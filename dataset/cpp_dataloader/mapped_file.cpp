#include "mapped_file.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


//------------------------------------------------------------------------------
// MappedFileReader

bool MappedFileReader::Open(const std::string& name) {
    shm_fd_ = shm_open(name.c_str(), O_RDONLY, 0);
    if (shm_fd_ == -1) {
        return false;
    }

    struct stat sb;
    if (fstat(shm_fd_, &sb) == -1) {
        close(shm_fd_);
        return false;
    }

    size_ = sb.st_size;
    data_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, shm_fd_, 0);
    return (data_ != MAP_FAILED);
}

void MappedFileReader::Close() {
    if (data_) munmap((void*)data_, size_);
    if (shm_fd_ != -1) close(shm_fd_);
}
