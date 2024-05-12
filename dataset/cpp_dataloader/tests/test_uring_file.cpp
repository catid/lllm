#include "uring_file.hpp"
#include "tools.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>

const char* kTestFile = "test_file.bin";
const int kNumReads = 1000;
const int kReadBytes = 4096;
const size_t kTestFileSize = kReadBytes * kNumReads;

std::hash<std::string> hasher;
std::vector<size_t> hashes;

void create_test_file() {
    std::cout << "Creating test file..." << std::endl;
    FILE* file = fopen(kTestFile, "wb");
    if (file == nullptr) {
        std::cerr << "Failed to create test file" << std::endl;
        exit(-1);
    }

    for (size_t i = 0; i < kNumReads; ++i) {
        std::string data;
        data.reserve(kReadBytes);

        for (size_t j = 0; j < kReadBytes; ++j) {
            char byte = static_cast<char>(rand() % 256);
            data.push_back(byte);
        }

        fwrite(data.data(), 1, data.size(), file);
        hashes.push_back(hasher(data));
    }

    fclose(file);
    std::cout << "Test file created successfully" << std::endl;
}

void read_callback(ssize_t res, uint8_t* buffer, void* user_data) {
    size_t index = *reinterpret_cast<size_t*>(user_data);

    if (res != kReadBytes) {
        std::cerr << "Read size mismatch for request " << index << ". Expected: "
                  << kReadBytes << ", Actual: " << res << " (" << strerror(-res) << ")" << std::endl;
        exit(-1);
    } else {
        //std::cout << "Read completed for request " << index << ". Bytes read: " << res << std::endl;
    }

    std::string data(reinterpret_cast<char*>(buffer), res);
    size_t expected_hash = hashes[index];
    size_t actual_hash = hasher(data);

    if (actual_hash != expected_hash) {
        std::cerr << "Read data mismatch for request " << index << ". Expected hash: "
                  << expected_hash << ", Actual hash: " << actual_hash << std::endl;
        exit(-1);
    }

    //std::cout << "Read data verified successfully for request " << index << std::endl;
}

bool RunTest() {
    create_test_file();

    std::cout << "Creating AsyncUringReader..." << std::endl;
    AsyncUringReader reader;
    if (!reader.Open(kTestFile, kReadBytes, 16)) {
        std::cerr << "Failed to open AsyncUringReader" << std::endl;
        return false;
    }

    int64_t t0 = GetNsec();

    std::vector<size_t> indices(kNumReads);
    for (size_t i = 0; i < kNumReads; ++i) {
        indices[i] = i;
        //std::cout << "Submitting read request " << i << "..." << std::endl;
        bool success = reader.Read(
            i * kReadBytes,
            kReadBytes,
            read_callback,
            &indices[i]);
        if (!success) {
            std::cerr << "Failed to submit read request " << i << std::endl;
            reader.Close();
            return false;
        }
    }

    reader.Close();

    int64_t t1 = GetNsec();

    std::cout << "Read " << kNumReads << " blocks in " << (t1 - t0) / 1000000.0 << " ms" << std::endl;

    return true;
}

int main() {
    bool success = RunTest();

    remove(kTestFile);

    if (!success) {
        std::cerr << "Test failed" << std::endl;
        return -1;
    }

    std::cout << "All tests passed" << std::endl;
    return 0;
}
