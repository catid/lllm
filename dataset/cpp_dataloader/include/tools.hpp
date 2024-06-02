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

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& getInstance();

    Logger();
    ~Logger();

    void SetLogLevel(LogLevel level);
    void SetCallback(std::function<void(LogLevel, const std::string&)> callback);

    class LogStream {
    public:
        LogStream(Logger& logger, LogLevel level) : logger_(logger), level_(level) {}
        LogStream(const LogStream&& other) : logger_(other.logger_), level_(other.level_) {}
        ~LogStream() {
            logger_.Log(level_, std::move(ss_));
        }

        template<typename T>
        LogStream& operator<<(const T& value) {
            ss_ << value;
            return *this;
        }

    private:
        Logger& logger_;
        LogLevel level_;
        std::ostringstream ss_;
    };

    LogStream Debug() { return LogStream(*this, LogLevel::DEBUG); }
    LogStream Info() { return LogStream(*this, LogLevel::INFO); }
    LogStream Warn() { return LogStream(*this, LogLevel::WARN); }
    LogStream Error() { return LogStream(*this, LogLevel::ERROR); }

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
    LogLevel CurrentLogLevel;
    std::function<void(LogLevel, const std::string&)> Callback;

    std::atomic<bool> Terminated;

    void Log(LogLevel level, std::ostringstream&& message);
    void RunLogger();
    void ProcessLogQueue();
};

// Macros for logging
#define LOG_DEBUG() Logger::getInstance().Debug()
#define LOG_INFO() Logger::getInstance().Info()
#define LOG_WARN() Logger::getInstance().Warn()
#define LOG_ERROR() Logger::getInstance().Error()


//------------------------------------------------------------------------------
// TokenizedAllocator

struct TokenizedBuffer {
    std::vector<uint32_t> text;
};

class TokenizedAllocator {
public:
    std::shared_ptr<TokenizedBuffer> Allocate(
        const uint32_t* tokenized_text,
        uint32_t text_length);
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
