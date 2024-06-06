#pragma once

#include <string>
#include <cstdint>


//------------------------------------------------------------------------------
// MappedFileReader

class MappedFileReader {
public:
    ~MappedFileReader() {
        Close();
    }

    bool Open(const std::string& name);
    void Close();
    const uint8_t* GetData() const {
        return reinterpret_cast<const uint8_t*>(data_);
    }
    size_t GetSize() const {
        return size_;
    }
    bool IsValid() const {
        return is_valid_;
    }

private:
    const void* data_ = nullptr;
    size_t size_ = 0;
    bool is_valid_ = false;
    int fd_ = -1;
};
