#include "tools.hpp"

#include <cstring>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif


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
