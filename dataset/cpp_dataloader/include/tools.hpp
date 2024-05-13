#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>


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
