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
