#include "uring_file.hpp"

#include "tools.hpp"

#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/statvfs.h>

#include <fstream>


//------------------------------------------------------------------------------
// Aligned Malloc

static void* arb_align_malloc(size_t alignment, size_t size) {
    // Ensure alignment is greater than zero
    if (alignment == 0) {
        return nullptr;
    }

    // Allocate enough memory to ensure we can find an aligned block
    // Add alignment to account for alignment adjustment and sizeof(void*) to store the original pointer
    size_t extra_space = alignment + sizeof(void*);
    void* original = std::calloc(size + extra_space, 1);

    if (!original) {
        return nullptr; // Allocation failed
    }

    // Find the aligned memory address
    uintptr_t raw_address = reinterpret_cast<uintptr_t>(original) + sizeof(void*);
    uintptr_t aligned_address = (raw_address + alignment - 1) / alignment * alignment;

    // Store the original pointer just before the aligned address
    reinterpret_cast<void**>(aligned_address)[-1] = original;

    return reinterpret_cast<void*>(aligned_address);
}

static void arb_align_free(void* ptr) {
    if (ptr) {
        // Retrieve the original pointer stored before the aligned address
        void* original = reinterpret_cast<void**>(ptr)[-1];
        std::free(original);
    }
}


//------------------------------------------------------------------------------
// IoReuseAllocator

io_data::~io_data() {
    arb_align_free(buffer);
}

void IoReuseAllocator::SetAlignBytes(int align_bytes)
{
    Clear();

    align_bytes_ = align_bytes;
}

std::shared_ptr<io_data> IoReuseAllocator::Allocate(int bytes) {
    std::lock_guard<std::mutex> lock(Lock);

    std::shared_ptr<io_data> data;

    if (!Freed.empty()) {
        data = Freed.back();
        Freed.pop_back();
    } else {
        data = std::make_shared<io_data>();
    }

    if (!data->buffer || data->buffer_bytes < bytes) {
        arb_align_free(data->buffer);

        data->buffer = (uint8_t*)arb_align_malloc(align_bytes_, bytes);
        if (!data->buffer) {
            return nullptr;
        }
        data->buffer_bytes = bytes;
    }

    Used.push_back(data);

    data->RefCount++;
    data->callback = nullptr;
    return data;
}

void IoReuseAllocator::Free(io_data* data) {
    if (!data) {
        return;
    }

    std::lock_guard<std::mutex> lock(Lock);

    for (auto it = Used.begin(); it != Used.end(); ++it) {
        if (it->get() == data) {
            data->callback = nullptr;
            data->RefCount--;
            if (data->RefCount != 0) {
                LOG_ERROR() << "IoReuseAllocator: RefCount != 0 on freed data";
                return;
            }

            Freed.push_back(*it);

            Used.erase(it);

            return;
        }
    }

    LOG_ERROR() << "Failed to free io_data: Not found in Used list";
}

void IoReuseAllocator::Clear() {
    std::lock_guard<std::mutex> lock(Lock);

    Freed.clear();
    Used.clear();
}


//------------------------------------------------------------------------------
// FileEndCache

bool FileEndCache::FillCache(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG_ERROR() << "FileEndCache: Failed to open file: " << file_path;
        return false;
    }

    FileBytes = file.tellg();

    FinalBytes = std::min((uint64_t)kFileEndCacheBytes, FileBytes);
    FinalOffset = FileBytes - FinalBytes;

    file.seekg(FinalOffset);
    if (!file) {
        LOG_ERROR() << "FileEndCache: Failed to seek to position: " << FinalOffset;
        return false;
    }

    file.read(reinterpret_cast<char*>(FinalBuffer), FinalBytes);
    if (!file) {
        LOG_ERROR() << "FileEndCache: Failed to read " << FinalBytes << " bytes from file";
        return false;
    }

    return true;
}

bool FileEndCache::IsFullySatisfied(uint64_t offset, uint32_t bytes, ReadCallback callback)
{
    if (offset >= FinalOffset) {
        callback(FinalBuffer + offset - FinalOffset, bytes);
        return true;
    }
    return false;
}


//------------------------------------------------------------------------------
// AsyncUringReader

bool AsyncUringReader::Open(
    const std::string& file_path,
    int queue_depth)
{
    // Align allocations to the block size of the file device
    struct statvfs buf;
    int block_size = 4096; // Good default for most block devices
    if (statvfs(file_path.c_str(), &buf) == 0) {
        block_size = buf.f_bsize;
    }

    Allocator.SetAlignBytes(block_size);

    fd_ = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
        perror("AsyncUringReader: open failed");
        return false;
    }

    if (!EndCache.FillCache(file_path)) {
        return false;
    }

    posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);

    if (io_uring_queue_init(queue_depth, &ring_, 0) < 0) {
        perror("AsyncUringReader: io_uring_queue_init failed");
        return false;
    }
    ring_initialized_ = true;

    InflightCount = 0;

    Terminated = false;
    Thread = std::make_shared<std::thread>(&AsyncUringReader::Loop, this);

    return true;
}

void AsyncUringReader::Close() {
    {
        std::unique_lock<std::mutex> lock(Lock);
        Terminated = true;
        Condition.notify_one();
    }
    JoinThread(Thread);

    while (InflightCount > 0) {
        HandleNext();
    }

    if (ring_initialized_) {
        io_uring_queue_exit(&ring_);
        ring_ = io_uring{};
    }

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    if (InflightCount > 0) {
        LOG_ERROR() << "AsyncUringReader: Not all inflight requests were completed.  InflightCount=" << InflightCount;
    }
}

io_data* AsyncUringReader::HandleCqe(struct io_uring_cqe* cqe) {
    io_data* data = static_cast<io_data*>(io_uring_cqe_get_data(cqe));

    if (!data) {
        LOG_ERROR() << "AsyncUringReader: Invalid data pointer";
        return nullptr;
    }

    if (cqe->res < 0 || cqe->res + data->app_offset < data->app_bytes) {
        LOG_ERROR() << "AsyncUringReader: Error in callback.  res = " << cqe->res;
        data->callback(nullptr, 0);
    } else {
        data->callback(data->buffer + data->app_offset, data->app_bytes);
    }

    InflightCount--;

    return data;
}

bool AsyncUringReader::Read(
    uint64_t offset,
    uint32_t bytes,
    ReadCallback callback)
{
    if (EndCache.IsFullySatisfied(offset, bytes, callback)) {
        return true;
    }

    // Round offset down to the nearest 512-byte block
    // Then round read length up to the nearest 512-byte block
    const uint32_t app_offset = (uint32_t)offset % 512;
    const uint64_t request_offset = offset - app_offset;
    const uint32_t request_blocks = (bytes + app_offset + 512 - 1) / 512;
    const uint32_t request_bytes = request_blocks * 512;

    auto data = Allocator.Allocate(request_bytes);
    if (!data) {
        return false;
    }
    data->callback = callback;
    data->request_bytes = request_bytes;
    data->app_offset = app_offset;
    data->app_bytes = bytes;

    {
        std::unique_lock<std::mutex> lock(SubmitLock);

        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
        if (!sqe) {
            return false;
        }

        io_uring_prep_read(sqe, fd_, data->buffer, request_bytes, request_offset);
        io_uring_sqe_set_data(sqe, data.get());

        int r = io_uring_submit(&ring_);
        if (r < 0) {
            LOG_ERROR() << "AsyncUringReader: io_uring_submit failed: " << r << " (" << strerror(-r) << ")";
            return false;
        }
    }

    bool needs_notify = false;
    {
        std::unique_lock<std::mutex> lock(Lock);
        if (InflightCount++ <= 0) {
            needs_notify = true;
        }
    }
    if (needs_notify) {
        Condition.notify_one();
    }

    return true;
}

void AsyncUringReader::Loop() {
    while (!Terminated) {
        // Wait until termination or completions
        {
            std::unique_lock<std::mutex> lock(Lock);
            Condition.wait(lock, [this]{ return Terminated || InflightCount > 0; });
        }

        while (!Terminated && InflightCount > 0) {
            HandleNext();
        }
    }
}

void AsyncUringReader::HandleNext()
{
    struct io_uring_cqe* cqe = nullptr;
    io_uring_wait_cqe(&ring_, &cqe);

    io_data* data = HandleCqe(cqe);

    io_uring_cqe_seen(&ring_, cqe);

    Allocator.Free(data);
}
