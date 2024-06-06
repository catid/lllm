#include "mapped_file.hpp"

#include "tools.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


//------------------------------------------------------------------------------
// MappedFileReader

bool MappedFileReader::Open(const std::string& name) {
    fd_ = open(name.c_str(), O_RDONLY);
    if (fd_ == -1) {
        LOG_ERROR() << "MappedFileReader: Failed to open file: " << name;
        return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        LOG_ERROR() << "MappedFileReader: Failed to stat file: " << name;
        close(fd_);
        return false;
    }

    size_ = sb.st_size;
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    return (data_ != MAP_FAILED);
}

void MappedFileReader::Close() {
    if (data_) {
        munmap((void*)data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}
