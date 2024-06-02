#include "tools.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <cstring>
#include <chrono>
#include <iostream>


//------------------------------------------------------------------------------
// Logger

// Initialize static members
std::unique_ptr<Logger> Logger::Instance;
std::once_flag Logger::InitInstanceFlag;

Logger& Logger::getInstance() {
    std::call_once(InitInstanceFlag, []() {
        Instance.reset(new Logger);
    });
    return *Instance;
}

Logger::Logger() : CurrentLogLevel(LogLevel::INFO), Terminated(false) {
    LoggerThread = std::thread(&Logger::RunLogger, this);
}

Logger::~Logger() {
    Terminate();
    if (LoggerThread.joinable()) {
        LoggerThread.join();
    }
    ProcessLogQueue(); // Process any remaining logs
}

void Logger::SetLogLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(LogQueueMutex);
    CurrentLogLevel = level;
}

void Logger::SetCallback(std::function<void(LogLevel, const std::string&)> callback) {
    Callback = callback;
}

void Logger::Log(LogLevel level, std::ostringstream&& message) {
    std::lock_guard<std::mutex> lock(LogQueueMutex);
    LogQueue.push_back({level, message.str()});
    LogQueueCV.notify_one();
}

void Logger::Terminate() {
    Terminated = true;
    LogQueueCV.notify_one();
}

void Logger::RunLogger() {
    while (!Terminated) {
        std::unique_lock<std::mutex> lock(LogQueueMutex);
        LogQueueCV.wait(lock, [this] { return !LogQueue.empty() || Terminated; });

        if (!LogQueue.empty()) {
            ProcessLogQueue();
        }

        if (Terminated) {
            break;
        }
    }
}

void Logger::ProcessLogQueue() {
    std::vector<LogEntry> logsToProcess;
    logsToProcess.swap(LogQueue);

    for (const auto& entry : logsToProcess) {
        if (entry.Level >= CurrentLogLevel) {
            if (Callback) {
                Callback(entry.Level, entry.Message);
            } else {
                switch (entry.Level) {
                    case LogLevel::DEBUG:
                        std::cout << "[DEBUG] " << entry.Message << std::endl;
                        break;
                    case LogLevel::INFO:
                        std::cout << "[INFO] " << entry.Message << std::endl;
                        break;
                    case LogLevel::WARN:
                        std::cerr << "[WARN] " << entry.Message << std::endl;
                        break;
                    case LogLevel::ERROR:
                        std::cerr << "[ERROR] " << entry.Message << std::endl;
                        break;
                }
            }
        }
    }
}


//------------------------------------------------------------------------------
// TokenizedAllocator

std::shared_ptr<TokenizedBuffer> TokenizedAllocator::Allocate(
    const uint32_t* tokenized_text,
    uint32_t text_length)
{
    std::shared_ptr<TokenizedBuffer> buffer;

    if (free_buffers_count_ > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!free_buffers_.empty()) {
            buffer = free_buffers_.back();
            free_buffers_.pop_back();
            free_buffers_count_--;
        }
    }

    if (!buffer) {
        buffer = std::make_shared<TokenizedBuffer>();
    }

    buffer->text.resize(text_length);
    memcpy(buffer->text.data(), tokenized_text, text_length * sizeof(uint32_t));

    return buffer;
}

void TokenizedAllocator::Free(std::shared_ptr<TokenizedBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers_.push_back(buffer);
    free_buffers_count_++;
}


//------------------------------------------------------------------------------
// Tools

void JoinThread(std::shared_ptr<std::thread> th)
{
    if (!th || !th->joinable()) {
        return;
    }

    try {
        th->join();
    } catch (...) {
        return;
    }
}

int64_t GetNsec()
{
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return static_cast<int64_t>(tp.tv_sec) * 1000000000LL + tp.tv_nsec;
}

void set_thread_affinity(int cpu_id) {
    // Set thread affinity
    #ifdef _WIN32
        HANDLE thread_handle = GetCurrentThread();
        DWORD_PTR thread_affinity_mask = 1LL << cpu_id;
        SetThreadAffinityMask(thread_handle, thread_affinity_mask);
    #else
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        CPU_SET(cpu_id, &cpu_set);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    #endif
}
