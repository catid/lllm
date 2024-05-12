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

#include <iostream>


//------------------------------------------------------------------------------
// IoReuseAllocator

void IoReuseAllocator::SetAlignBytes(int align_bytes)
{
    align_bytes_ = align_bytes;
}

std::shared_ptr<io_data> IoReuseAllocator::Allocate(int bytes) {
    std::lock_guard<std::mutex> lock(Lock);

    std::shared_ptr<io_data> data;

    if (!Freed.empty()) {
        data = Freed.back();
        Freed.pop_back();
        if (data->buffer_bytes < bytes) {
            free(data->buffer);
            data->buffer = nullptr;
        }
    } else {
        data = std::make_shared<io_data>();
    }

    if (!data->buffer) {
        data->buffer = (uint8_t*)aligned_alloc(align_bytes_, bytes);
        if (!data->buffer) {
            return nullptr;
        }
    }

    Used.push_back(data);

    data->buffer_bytes = bytes;
    data->callback = nullptr;
    return data;
}

void IoReuseAllocator::Free(io_data* data) {
    std::lock_guard<std::mutex> lock(Lock);

    for (auto it = Used.begin(); it != Used.end(); ++it) {
        if (it->get() == data) {
            auto shared_ptr = *it;
            Freed.push_back(shared_ptr);
            Used.erase(it);
            data->callback = nullptr;
            return;
        }
    }

    std::cerr << "Failed to free io_data: Not found in Used list" << std::endl;
}


//------------------------------------------------------------------------------
// AsyncUringReader

bool AsyncUringReader::Open(
    const char* filename,
    int queue_depth)
{
    // Align allocations to the block size of the file device
    struct statvfs buf;
    int block_size = 4096; // Good default for most block devices
    if (statvfs(filename, &buf) == 0) {
        block_size = buf.f_bsize;
    }

    Allocator.SetAlignBytes(block_size);

    fd = open(filename, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("AsyncUringReader: open failed");
        return false;
    }

    if (io_uring_queue_init(queue_depth, &ring, 0) < 0) {
        perror("AsyncUringReader: io_uring_queue_init failed");
        return false;
    }
    ring_initialized = true;

    inflight = 0;

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

    while (inflight > 0) {
        struct io_uring_cqe* cqe = nullptr;
        io_uring_wait_cqe(&ring, &cqe);
        HandleCqe(cqe);
        io_uring_cqe_seen(&ring, cqe);
    }

    if (ring_initialized) {
        io_uring_queue_exit(&ring);
        ring = io_uring{};
    }

    if (fd >= 0) {
        close(fd);
    }

    if (inflight > 0) {
        std::cerr << "AsyncUringReader: Not all inflight requests were completed.  inflight = " << inflight << std::endl;
    }
}

void AsyncUringReader::HandleCqe(struct io_uring_cqe* cqe) {
    io_data* data = static_cast<io_data*>(io_uring_cqe_get_data(cqe));

    if (!data) {
        std::cerr << "AsyncUringReader: Invalid data pointer" << std::endl;
        return;
    }

    if (cqe->res < 0 || cqe->res + data->app_offset < data->app_bytes) {
        std::cerr << "AsyncUringReader: Error in callback.  res = " << cqe->res << std::endl;
        data->callback(nullptr, 0);
    } else {
        data->callback(data->buffer + data->app_offset, data->app_bytes);
    }

    Allocator.Free(data);

    inflight--;
}

bool AsyncUringReader::Read(
    uint64_t offset,
    uint32_t bytes,
    ReadCallback callback)
{
    // Round offset down to the nearest 512-byte block
    // Then round read length up to the nearest 512-byte block
    uint32_t app_offset = (uint32_t)offset & 511;
    uint64_t request_offset = offset - app_offset;
    uint32_t request_bytes = ((bytes + app_offset + 511) & ~511);

    auto data = Allocator.Allocate(request_bytes);
    if (!data) {
        return false;
    }
    data->callback = callback;
    data->request_bytes = request_bytes;
    data->app_offset = app_offset;
    data->app_bytes = bytes;

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
        return false;
    }

    io_uring_prep_read(sqe, fd, data->buffer, request_bytes, request_offset);
    io_uring_sqe_set_data(sqe, data.get());
    int r = io_uring_submit(&ring);
    if (r < 0) {
        std::cerr << "AsyncUringReader: io_uring_submit failed: " << r << " (" << strerror(-r) << ")" << std::endl;
        return false;
    }

    bool needs_notify = false;
    {
        std::unique_lock<std::mutex> lock(Lock);
        if (inflight++ <= 0) {
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
            Condition.wait(lock, [this]{ return Terminated || inflight > 0; });
        }

        while (inflight > 0) {
            if (Terminated) {
                break;
            }

            struct io_uring_cqe* cqe = nullptr;
            io_uring_wait_cqe(&ring, &cqe);

            HandleCqe(cqe);

            io_uring_cqe_seen(&ring, cqe);
        }
    }
}
