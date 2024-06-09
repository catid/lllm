#include "uring_file.hpp"
#include "tools.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>

#include <city.h>

const char* kTestFile = "test_file.bin";
const int kMinReads = 1000000;
const int kMaxReadBytes = 2048;
const size_t kMinTestFileSize = (size_t)kMaxReadBytes * (size_t)kMinReads / 2;

std::vector<uint64_t> hashes;
std::vector<uint32_t> read_sizes;
std::vector<uint64_t> read_offsets;
int total_reads = 0;
std::atomic<int> num_errors(0);

bool create_test_file(uint64_t& total_bytes) {
    LOG_INFO() << "Creating test file...";

    std::ofstream file(kTestFile, std::ios::binary);
    if (!file) { 
        LOG_ERROR() << "Failed to create test file";
        return false;
    }

    uint64_t last_log_bytes = 0;
    total_bytes = 0;
    uint8_t buffer[kMaxReadBytes];

    read_sizes.clear();
    hashes.clear();
    total_reads = 0;

    while (total_bytes < kMinTestFileSize && total_reads < kMinReads) {
        int read_size = rand() % kMaxReadBytes + 1;

        for (int j = 0; j < read_size; ++j) {
            buffer[j] = static_cast<uint8_t>(rand() % 256);
        }

        if (!file.write(reinterpret_cast<const char*>(buffer), read_size)) {
            LOG_ERROR() << "Failed to write to test file";
            return false;
        }

        hashes.push_back(CityHash64(reinterpret_cast<const char*>(buffer), read_size));

        read_sizes.push_back(read_size);
        read_offsets.push_back(total_bytes);

        total_bytes += read_size;
        ++total_reads;

        if (total_bytes - last_log_bytes > 10 * 1024 * 1024) {
            last_log_bytes = total_bytes;
            LOG_INFO() << "Wrote " << total_bytes / 1000000.0 << " MB...";
        }
    }

    LOG_INFO() << "Created test file with " << total_reads << " reads and " << total_bytes << " bytes";
    return true;
}

bool RunTest() {
    uint64_t test_file_bytes = 0;
    if (!create_test_file(test_file_bytes)) {
        return false;
    }

    LOG_INFO() << "Performing random-access reads and verifying data...";
    AsyncUringReader reader;
    if (!reader.Open(kTestFile)) {
        LOG_ERROR() << "Failed to open AsyncUringReader";
        return false;
    }

    if (reader.GetSize() != test_file_bytes) {
        LOG_ERROR() << "AsyncUringReader size mismatch: expected " << test_file_bytes << ", found " << reader.GetSize();
        return false;
    }

    std::vector<uint32_t> read_indices(total_reads);
    for (int i = 0; i < total_reads; ++i) {
        read_indices[i] = i;
    }
    std::shuffle(read_indices.begin(), read_indices.end(), std::mt19937(std::random_device{}()));

    std::atomic<uint64_t> total_bytes = ATOMIC_VAR_INIT(0);

    int64_t t0 = GetNsec();
    for (int j = 0; j < total_reads; ++j) {
        uint32_t index = read_indices[j];
        uint32_t expected_bytes = read_sizes[index];
        uint64_t read_offset = read_offsets[index];

        bool success = reader.Read(read_offset, expected_bytes,
            [index, read_offset, expected_bytes, &total_bytes](uint8_t* data, uint32_t bytes)
        {
            if (!data || bytes == 0) {
                LOG_ERROR() << "Read error for request " << index << " (offset: " << read_offset << ", bytes: " << expected_bytes << ")";
                ++num_errors;
                return;
            }

            if (bytes != expected_bytes) {
                LOG_ERROR() << "Read size mismatch for request " << index << ". Expected: " << expected_bytes << ", Actual: " << bytes;
                ++num_errors;
                return;
            }

            uint64_t expected_hash = hashes[index];
            uint64_t actual_hash = CityHash64(reinterpret_cast<char*>(data), bytes);
            if (actual_hash != expected_hash) {
                LOG_ERROR() << "Read data mismatch for request " << index << ". Expected hash: " << expected_hash << ", Actual hash: " << actual_hash;
                ++num_errors;
                return;
            }

            total_bytes += bytes;
        });
        if (!success) {
            LOG_ERROR() << "Failed to submit read request " << index;
            ++num_errors;
            break;
        }
    }

    // We do a busy loop here for precise timing but normally we would not do this.
    while (reader.IsBusy()) {
        std::this_thread::yield();
    }
    reader.Close();

    int64_t t1 = GetNsec();
    if (num_errors > 0) {
        LOG_ERROR() << "Test failed with " << num_errors << " errors";
        return false;
    }

    LOG_INFO() << "Read " << total_reads << " blocks in " << (t1 - t0) / 1000000.0
        << " ms: " << (total_bytes * 1000.0 / (t1 - t0)) << " MB/s";
    return true;
}

int main() {
    bool success = RunTest();
    remove(kTestFile);
    if (!success) {
        LOG_ERROR() << "Test failed";
        return -1;
    }
    LOG_INFO() << "All tests passed";
    return 0;
}
