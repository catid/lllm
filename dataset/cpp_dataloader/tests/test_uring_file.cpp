#include "uring_file.hpp"
#include "tools.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <city.h>

const char* kTestFile = "test_file.bin";
const int kNumReads = 100000;
const int kMaxReadBytes = 2048;
const size_t kMinTestFileSize = kMaxReadBytes * kNumReads / 2;

std::vector<uint64_t> hashes;
std::vector<size_t> read_sizes;
std::atomic<int> num_errors(0);

bool create_test_file() {
    std::cout << "Creating test file..." << std::endl;

    std::ofstream file(kTestFile, std::ios::binary);
    if (!file) { 
        std::cerr << "Failed to create test file" << std::endl;
        return false;
    }

    size_t total_bytes = 0;
    uint8_t buffer[kMaxReadBytes];

    while (total_bytes < kMinTestFileSize)
    {
        size_t read_size = rand() % kMaxReadBytes + 1;

        for (size_t j = 0; j < read_size; ++j) {
            buffer[j] = static_cast<uint8_t>(rand() % 256);
        }

        if (!file.write(reinterpret_cast<const char*>(buffer), read_size)) {
            std::cerr << "Failed to write to test file" << std::endl;
            return false;
        }

        hashes.push_back(CityHash64(reinterpret_cast<const char*>(buffer), read_size));

        read_sizes.push_back(read_size);
        total_bytes += read_size;
    }

    std::cout << "Test file created successfully" << std::endl;
    return true;
}

bool RunTest() {
    create_test_file();

    std::cout << "Creating AsyncUringReader..." << std::endl;
    AsyncUringReader reader;
    if (!reader.Open(kTestFile)) {
        std::cerr << "Failed to open AsyncUringReader" << std::endl;
        return false;
    }

    int64_t t0 = GetNsec();
    size_t offset = 0;
    for (size_t i = 0; i < kNumReads; ++i) {
        //std::cout << "Submitting read request " << i << "..." << std::endl;
        bool success = reader.Read(offset, read_sizes[i],
            [i](uint8_t* data, uint32_t bytes)
        {
            if (!data || bytes == 0) {
                std::cerr << "Read error for request " << i << std::endl;
                ++num_errors;
                return;
            }

            size_t expected_size = read_sizes[i];
            if (bytes != expected_size) {
                std::cerr << "Read size mismatch for request " << i << ". Expected: " << expected_size << ", Actual: " << bytes << std::endl;
                ++num_errors;
                return;
            }

            uint64_t expected_hash = hashes[i];
            uint64_t actual_hash = CityHash64(reinterpret_cast<char*>(data), bytes);
            if (actual_hash != expected_hash) {
                std::cerr << "Read data mismatch for request " << i << ". Expected hash: " << expected_hash << ", Actual hash: " << actual_hash << std::endl;
                ++num_errors;
                return;
            }
        });
        if (!success) {
            std::cerr << "Failed to submit read request " << i << std::endl;
            ++num_errors;
            break;
        }
        offset += read_sizes[i];
    }

    while (reader.IsBusy()) {
        // Wait for all reads to complete
    }
    reader.Close();

    int64_t t1 = GetNsec();
    if (num_errors > 0) {
        std::cerr << "Test failed with " << num_errors << " errors" << std::endl;
        return false;
    }

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
