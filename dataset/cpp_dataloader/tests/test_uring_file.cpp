#include "uring_file.hpp"
#include "tools.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>

#include <city.h>

const char* kTestFile = "test_file.bin";
const int kMinReads = 1000000;
const int kMaxReadBytes = 2048;
const size_t kMinTestFileSize = kMaxReadBytes * kMinReads / 2;

std::vector<uint64_t> hashes;
std::vector<uint32_t> read_sizes;
std::vector<uint32_t> read_offsets;
int total_reads = 0;
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

    read_sizes.clear();
    hashes.clear();
    total_reads = 0;

    while (total_bytes < kMinTestFileSize && total_reads < kMinReads) {
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
        read_offsets.push_back(total_bytes);

        total_bytes += read_size;
        ++total_reads;
    }

    std::cout << "Created test file with " << total_reads << " reads and " << total_bytes << " bytes" << std::endl;
    return true;
}

bool RunTest() {
    create_test_file();

    std::cout << "Performing random-access reads and verifying data..." << std::endl;
    AsyncUringReader reader;
    if (!reader.Open(kTestFile)) {
        std::cerr << "Failed to open AsyncUringReader" << std::endl;
        return false;
    }

    std::vector<uint32_t> read_indices(total_reads);
    for (int i = 0; i < total_reads; ++i) {
        read_indices[i] = i;
    }
    std::shuffle(read_indices.begin(), read_indices.end(), std::mt19937(std::random_device{}()));

    int64_t t0 = GetNsec();
    for (int j = 0; j < total_reads; ++j) {
        uint32_t index = read_indices[j];
        uint32_t expected_bytes = read_sizes[index];
        uint32_t read_offset = read_offsets[index];

        bool success = reader.Read(read_offset, expected_bytes,
            [index, read_offset, expected_bytes](uint8_t* data, uint32_t bytes)
        {
            if (!data || bytes == 0) {
                std::cerr << "Read error for request " << index << " (offset: " << read_offset << ", bytes: " << expected_bytes << ")" << std::endl;
                ++num_errors;
                return;
            }

            if (bytes != expected_bytes) {
                std::cerr << "Read size mismatch for request " << index << ". Expected: " << expected_bytes << ", Actual: " << bytes << std::endl;
                ++num_errors;
                return;
            }

            uint64_t expected_hash = hashes[index];
            uint64_t actual_hash = CityHash64(reinterpret_cast<char*>(data), bytes);
            if (actual_hash != expected_hash) {
                std::cerr << "Read data mismatch for request " << index << ". Expected hash: " << expected_hash << ", Actual hash: " << actual_hash << std::endl;
                ++num_errors;
                return;
            }
        });
        if (!success) {
            std::cerr << "Failed to submit read request " << index << std::endl;
            ++num_errors;
            break;
        }
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

    std::cout << "Read " << total_reads << " blocks in " << (t1 - t0) / 1000000.0 << " ms" << std::endl;
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
