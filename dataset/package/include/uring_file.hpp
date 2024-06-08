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
    ~io_data();

    int buffer_bytes = 0;
    uint8_t* buffer = nullptr;

    ReadCallback callback;
    uint32_t request_bytes = 0;
    uint32_t app_offset = 0;
    uint32_t app_bytes = 0;

    int RefCount = 0;
};

class IoReuseAllocator {
public:
    // Must be called before any other methods
    void SetAlignBytes(int align_bytes);

    std::shared_ptr<io_data> Allocate(int bytes);
    void Free(io_data* data);

    void Clear();

private:
    int align_bytes_ = 4096;

    std::mutex Lock;
    std::vector<std::shared_ptr<io_data>> Freed;
    std::vector<std::shared_ptr<io_data>> Used;
};


//------------------------------------------------------------------------------
// FileEndCache

static const int kFileEndCacheBytes = 4096;

/*
    This class caches the last kFileEndCacheBytes bytes of the file.
    It helps to avoid reading past the physical end of the file,
    which may help avoid OS bugs.
    My version of Linux io_uring handles this well already, but
    it's possible that other implementations don't.
*/
struct FileEndCache {
    bool FillCache(const std::string& file_path);

    // Invokes callback if fully satisfied and returns true.
    bool IsFullySatisfied(uint64_t offset, uint32_t bytes, ReadCallback callback);

    uint64_t FileBytes = 0;

    uint8_t FinalBuffer[kFileEndCacheBytes];
    uint64_t FinalOffset = 0;
    uint32_t FinalBytes = 0;
};


//------------------------------------------------------------------------------
// AsyncUringReader

class AsyncUringReader {
public:
    ~AsyncUringReader() {
        Close();
    }

    bool Open(
        const std::string& file_path,
        int queue_depth = 8);
    void Close();

    uint64_t GetSize() const { return EndCache.FileBytes; }

    // Returns false if the reader is busy.
    // This can invoke the callback during the Read() call if it is in cache.
    bool Read(
        uint64_t offset,
        uint32_t bytes,
        ReadCallback callback);

    bool IsBusy() const { return InflightCount > 0; }

private:
    struct io_uring ring_{};
    bool ring_initialized_ = false;

    int fd_ = -1;
    int block_size_ = 0;
    int read_align_bytes_ = 0;

    std::atomic<int> InflightCount = ATOMIC_VAR_INIT(0); 

    IoReuseAllocator Allocator;

    std::shared_ptr<std::thread> Thread;
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    std::mutex Lock;
    std::condition_variable Condition;

    FileEndCache EndCache;

    std::mutex SubmitLock;

    io_data* HandleCqe(struct io_uring_cqe* cqe);

    void Loop();
    void HandleNext();
};
