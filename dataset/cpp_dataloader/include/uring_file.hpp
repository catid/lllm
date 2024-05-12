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

#include <liburing.h>


//------------------------------------------------------------------------------
// IoReuseAllocator

// Passes errors to the callback: Negative values are errors
using ReadCallback = std::function<void(ssize_t bytes, uint8_t* buffer, void* user_data)>;

struct io_data {
    ~io_data() {
        if (buffer) {
            free(buffer);
        }
    }

    size_t length = 0;
    uint8_t* buffer = nullptr;

    off_t offset = 0;
    ReadCallback callback;
    void* user_data = nullptr;
};

class IoReuseAllocator {
public:
    IoReuseAllocator(int buffer_bytes = 4096);

    std::shared_ptr<io_data> Allocate();
    void Free(io_data* data);

private:
    int buffer_bytes_ = 0;

    std::mutex Lock;
    std::vector<std::shared_ptr<io_data>> Freed;
    std::vector<std::shared_ptr<io_data>> Used;
};


//------------------------------------------------------------------------------
// AsyncUringReader

class AsyncUringReader {
public:
    AsyncUringReader(int buffer_bytes = 4096)
        : Allocator(buffer_bytes)
    {}
    ~AsyncUringReader() {
        Close();
    }

    bool Open(const char* filename, int queue_depth);
    void Close();

    bool Read(
        off_t offset,
        size_t length,
        ReadCallback callback,
        void* user_data);

    bool IsBusy() const { return inflight > 0; }

private:
    struct io_uring ring{};
    bool ring_initialized = false;

    int fd = -1;
    std::atomic<int> inflight = ATOMIC_VAR_INIT(0); 

    IoReuseAllocator Allocator;

    std::shared_ptr<std::thread> Thread;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::mutex Lock;
    std::condition_variable Condition;

    void HandleCqe(struct io_uring_cqe* cqe);

    void Loop();
};
