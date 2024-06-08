#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <string>
#include <functional>
#include <sstream>


//------------------------------------------------------------------------------
// Logger

class Logger {
public:
    enum LogLevel { DEBUG, INFO, WARN, ERROR };

    static Logger& getInstance();

    Logger();
    ~Logger();

    void SetLogLevel(LogLevel level);
    void SetCallback(std::function<void(LogLevel, const std::string&)> callback);

    class LogStream {
    public:
        LogStream(Logger& logger, LogLevel level, bool log_enabled = true)
            : logger_(logger)
            , level_(level)
            , log_enabled_(log_enabled)
        {
        }
        LogStream(const LogStream&& other)
            : logger_(other.logger_)
            , level_(other.level_)
        {
        }
        ~LogStream() {
            if (log_enabled_) {
                logger_.Log(level_, std::move(ss_));
            }
        }

        template<typename T>
        LogStream& operator<<(const T& value) {
            if (log_enabled_) {
                ss_ << value;
            }
            return *this;
        }

    private:
        Logger& logger_;
        LogLevel level_;
        bool log_enabled_ = false;

        std::ostringstream ss_;
    };

    LogStream Debug() { return LogStream(*this, LogLevel::DEBUG, CurrentLogLevel <= LogLevel::DEBUG); }
    LogStream Info() { return LogStream(*this, LogLevel::INFO, CurrentLogLevel <= LogLevel::INFO); }
    LogStream Warn() { return LogStream(*this, LogLevel::WARN, CurrentLogLevel <= LogLevel::WARN); }
    LogStream Error() { return LogStream(*this, LogLevel::ERROR, CurrentLogLevel <= LogLevel::ERROR); }

    void Terminate();

private:
    static std::unique_ptr<Logger> Instance;
    static std::once_flag InitInstanceFlag;

    struct LogEntry {
        LogLevel Level;
        std::string Message;
    };

    std::vector<LogEntry> LogQueue;
    std::mutex LogQueueMutex;
    std::condition_variable LogQueueCV;

    std::thread LoggerThread;
    std::atomic<LogLevel> CurrentLogLevel = ATOMIC_VAR_INIT(LogLevel::INFO);
    std::function<void(LogLevel, const std::string&)> Callback;

    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::vector<LogEntry> LogsToProcess;

    void Log(LogLevel level, std::ostringstream&& message);
    void RunLogger();
    void ProcessLogQueue();
};

// Macros for logging
#define LOG_DEBUG() Logger::getInstance().Debug()
#define LOG_INFO() Logger::getInstance().Info()
#define LOG_WARN() Logger::getInstance().Warn()
#define LOG_ERROR() Logger::getInstance().Error()
#define LOG_TERMINATE() Logger::getInstance().Terminate();


//------------------------------------------------------------------------------
// TokenizedAllocator

struct TokenizedBuffer {
    std::vector<uint8_t> Data;
};

class TokenizedAllocator {
public:
    std::shared_ptr<TokenizedBuffer> Allocate(
        const void* data,
        uint32_t bytes);
    void Free(std::shared_ptr<TokenizedBuffer> buffer);

protected:
    std::mutex mutex_;
    std::vector<std::shared_ptr<TokenizedBuffer>> free_buffers_;
    std::atomic<uint32_t> free_buffers_count_ = ATOMIC_VAR_INIT(0);
};


//------------------------------------------------------------------------------
// Tools

void JoinThread(std::shared_ptr<std::thread> th);

int64_t GetNsec();

void set_thread_affinity(int cpu_id);


//------------------------------------------------------------------------------
// Serialization

inline void write_uint32_le(void* buffer, uint32_t value) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    ptr[0] = static_cast<uint8_t>(value);
    ptr[1] = static_cast<uint8_t>(value >> 8);
    ptr[2] = static_cast<uint8_t>(value >> 16);
    ptr[3] = static_cast<uint8_t>(value >> 24);
}

inline uint32_t read_uint32_le(const void* buffer) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    return static_cast<uint32_t>(ptr[0]) |
           (static_cast<uint32_t>(ptr[1]) << 8) |
           (static_cast<uint32_t>(ptr[2]) << 16) |
           (static_cast<uint32_t>(ptr[3]) << 24);
}

inline void write_uint64_le(void* buffer, uint64_t value) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    ptr[0] = static_cast<uint8_t>(value);
    ptr[1] = static_cast<uint8_t>(value >> 8);
    ptr[2] = static_cast<uint8_t>(value >> 16);
    ptr[3] = static_cast<uint8_t>(value >> 24);
    ptr[4] = static_cast<uint8_t>(value >> 32);
    ptr[5] = static_cast<uint8_t>(value >> 40);
    ptr[6] = static_cast<uint8_t>(value >> 48);
    ptr[7] = static_cast<uint8_t>(value >> 56);
}

inline uint64_t read_uint64_le(const void* buffer) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    return static_cast<uint64_t>(ptr[0]) |
           (static_cast<uint64_t>(ptr[1]) << 8) |
           (static_cast<uint64_t>(ptr[2]) << 16) |
           (static_cast<uint64_t>(ptr[3]) << 24) |
           (static_cast<uint64_t>(ptr[4]) << 32) |
           (static_cast<uint64_t>(ptr[5]) << 40) |
           (static_cast<uint64_t>(ptr[6]) << 48) |
           (static_cast<uint64_t>(ptr[7]) << 56);
}
