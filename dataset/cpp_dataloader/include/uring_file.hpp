#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

#include <functional>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>

#include <uring.h>


//------------------------------------------------------------------------------
// IoReuseAllocator

// On error passes nullptr buffer and negative bytes to the callback
using ReadCallback = std::function<void(uint8_t* buffer, uint32_t bytes)>;

struct io_data {
    ~io_data() {
        if (buffer) {
            free(buffer);
        }
    }

    int buffer_bytes = 0;
    uint8_t* buffer = nullptr;

    ReadCallback callback;
    uint32_t request_bytes = 0;
    uint32_t app_offset = 0;
    uint32_t app_bytes = 0;
};

class IoReuseAllocator {
public:
    void SetAlignBytes(int align_bytes);

    std::shared_ptr<io_data> Allocate(int bytes);
    void Free(io_data* data);

private:
    int align_bytes_ = 4096;

    std::mutex Lock;
    std::vector<std::shared_ptr<io_data>> Freed;
    std::vector<std::shared_ptr<io_data>> Used;
};


//------------------------------------------------------------------------------
// AsyncUringReader

class AsyncUringReader {
public:
    ~AsyncUringReader() {
        Close();
    }

    bool Open(
        const char* filename,
        int queue_depth = 8);
    void Close();

    // Returns false if the reader is busy
    bool Read(
        uint64_t offset,
        uint32_t bytes,
        ReadCallback callback);

    bool IsBusy() const { return inflight > 0; }

private:
    struct io_uring ring{};
    bool ring_initialized = false;

    int fd = -1;
    int block_size_ = 0;
    int read_align_bytes_ = 0;
    std::atomic<int> inflight = ATOMIC_VAR_INIT(0); 

    IoReuseAllocator Allocator;

    std::shared_ptr<std::thread> Thread;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::mutex Lock;
    std::condition_variable Condition;

    void HandleCqe(struct io_uring_cqe* cqe);

    void Loop();
};
