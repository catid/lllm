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

#include <iostream>


//------------------------------------------------------------------------------
// IoReuseAllocator

IoReuseAllocator::IoReuseAllocator(int buffer_bytes) {
    buffer_bytes_ = buffer_bytes;
}

std::shared_ptr<io_data> IoReuseAllocator::Allocate() {
    std::lock_guard<std::mutex> lock(Lock);

    std::shared_ptr<io_data> data;

    if (!Freed.empty()) {
        data = Freed.back();
        Freed.pop_back();
    } else {
        data = std::make_shared<io_data>();
        data->buffer = (uint8_t*)aligned_alloc(4096, buffer_bytes_);
        Used.push_back(data);
    }

    data->length = buffer_bytes_;
    data->callback = nullptr;
    data->user_data = nullptr;
    data->offset = 0;
    return data;
}

void IoReuseAllocator::Free(io_data* data) {
    data->callback = nullptr;

    std::lock_guard<std::mutex> lock(Lock);

    for (auto it = Used.begin(); it != Used.end(); ++it) {
        if (it->get() == data) {
            auto shared_ptr = *it;
            Used.erase(it);
            Freed.push_back(shared_ptr);
            break;
        }
    }
}


//------------------------------------------------------------------------------
// AsyncUringReader

bool AsyncUringReader::Open(const char* filename, int queue_depth) {
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

    if (ring_initialized) {
        io_uring_queue_exit(&ring);
        ring = io_uring{};
    }

    if (fd >= 0) {
        close(fd);
    }
}

void AsyncUringReader::HandleCqe(struct io_uring_cqe* cqe) {
    io_data* data = static_cast<io_data*>(io_uring_cqe_get_data(cqe));

    // Passes errors to the callback: Negative values are errors
    data->callback(cqe->res, data->buffer, data->user_data);

    Allocator.Free(data);

    inflight--;
}

bool AsyncUringReader::Read(
    off_t offset,
    size_t length,
    ReadCallback callback,
    void* user_data)
{
    auto data = Allocator.Allocate();
    if (data->length < length) {
        std::cerr << "AsyncUringReader: buffer too small" << std::endl;
        return false;
    }
    data->length = length;
    data->offset = offset;
    data->callback = callback;
    data->user_data = user_data;

    struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
    if (!sqe) {
        return false;
    }

    io_uring_prep_read(sqe, fd, data->buffer, length, offset);
    io_uring_sqe_set_data(sqe, data.get());
    int r = io_uring_submit(&ring);
    if (r < 0) {
        std::cerr << "AsyncUringReader: io_uring_submit failed: " << r << std::endl;
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
