#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>

class AsyncDiskReader {
private:
    struct io_uring ring;
    int fd;
    int inflight;

    struct io_data {
        off_t offset;
        size_t length;
        void* buffer;
        void (*callback)(ssize_t, void*);
        void* user_data;
    };

    void handle_cqe(struct io_uring_cqe* cqe);

public:
    AsyncDiskReader(const char* filename, int queue_depth);
    ~AsyncDiskReader();

    void submit_read(off_t offset, size_t length, void (*callback)(ssize_t, void*), void* user_data);
    void wait_for_completions();
};
