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

    static void handle_cqe(struct io_uring_cqe* cqe) {
        struct io_data* data = static_cast<struct io_data*>(io_uring_cqe_get_data(cqe));
        ssize_t res = cqe->res;

        if (res < 0) {
            fprintf(stderr, "AsyncDiskReader: read error: %s\n", strerror(-res));
        }

        data->callback(res, data->user_data);
        free(data->buffer);
        free(data);
    }

public:
    AsyncDiskReader(const char* filename, int queue_depth) {
        fd = open(filename, O_RDONLY | O_DIRECT);
        if (fd < 0) {
            perror("AsyncDiskReader: open failed");
            exit(1);
        }

        if (io_uring_queue_init(queue_depth, &ring, 0) < 0) {
            perror("AsyncDiskReader: io_uring_queue_init failed");
            exit(1);
        }

        inflight = 0;
    }

    ~AsyncDiskReader() {
        io_uring_queue_exit(&ring);
        close(fd);
    }

    void submit_read(off_t offset, size_t length, void (*callback)(ssize_t, void*), void* user_data) {
        struct io_data* data = static_cast<struct io_data*>(malloc(sizeof(struct io_data)));
        data->offset = offset;
        data->length = length;
        data->buffer = aligned_alloc(4096, length);
        data->callback = callback;
        data->user_data = user_data;

        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, data->buffer, length, offset);
        io_uring_sqe_set_data(sqe, data);

        inflight++;
        if (inflight == 1) {
            io_uring_submit(&ring);
        }
    }

    void wait_for_completions() {
        struct io_uring_cqe* cqe;

        while (inflight > 0) {
            io_uring_wait_cqe(&ring, &cqe);
            handle_cqe(cqe);
            io_uring_cqe_seen(&ring, cqe);
            inflight--;
        }
    }
};
