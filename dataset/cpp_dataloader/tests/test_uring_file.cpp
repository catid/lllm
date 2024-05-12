#include "uring_file.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>

const char* kTestFile = "test_file.bin";
const size_t kTestFileSize = 8192;
const size_t kNumReads = 1000;

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
        data.reserve(kTestFileSize / kNumReads);

        for (size_t j = 0; j < kTestFileSize / kNumReads; ++j) {
            char byte = static_cast<char>(rand() % 256);
            data.push_back(byte);
        }

        fwrite(data.data(), 1, data.size(), file);
        hashes.push_back(hasher(data));
    }

    fclose(file);
    std::cout << "Test file created successfully" << std::endl;
}

void remove_test_file() {
    std::cout << "Removing test file..." << std::endl;
    if (remove(kTestFile) == 0) {
        std::cout << "Test file removed successfully" << std::endl;
    } else {
        std::cerr << "Failed to remove test file" << std::endl;
    }
}

void read_callback(ssize_t res, uint8_t* buffer, void* user_data) {
    size_t index = *reinterpret_cast<size_t*>(user_data);
    std::cout << "Read completed for request " << index << ". Bytes read: " << res << " (" << strerror(-res) << ")" << std::endl;

    if (res != kTestFileSize / kNumReads) {
        std::cerr << "Read size mismatch for request " << index << ". Expected: "
                  << kTestFileSize / kNumReads << ", Actual: " << res << " (" << strerror(-res) << ")" << std::endl;
        exit(-1);
    }

    std::string data(reinterpret_cast<char*>(buffer), res);
    size_t expected_hash = hashes[index];
    size_t actual_hash = hasher(data);

    if (actual_hash != expected_hash) {
        std::cerr << "Read data mismatch for request " << index << ". Expected hash: "
                  << expected_hash << ", Actual hash: " << actual_hash << std::endl;
        exit(-1);
    }

    std::cout << "Read data verified successfully for request " << index << std::endl;
}

int main() {
    create_test_file();

    std::cout << "Creating AsyncUringReader..." << std::endl;
    AsyncUringReader reader;
    if (!reader.Open(kTestFile, kNumReads)) {
        std::cerr << "Failed to open AsyncUringReader" << std::endl;
        remove_test_file();
        return -1;
    }

    std::vector<size_t> indices(kNumReads);
    for (size_t i = 0; i < kNumReads; ++i) {
        indices[i] = i;
        std::cout << "Submitting read request " << i << "..." << std::endl;
        if (!reader.Read(i * (kTestFileSize / kNumReads), kTestFileSize / kNumReads, read_callback, &indices[i])) {
            std::cerr << "Failed to submit read request " << i << std::endl;
            reader.Close();
            remove_test_file();
            return -1;
        }
    }

    std::cout << "Waiting for read completions..." << std::endl;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!reader.IsBusy()) {
            break;
        }
    }

    reader.Close();
    remove_test_file();
    std::cout << "All tests passed" << std::endl;
    return 0;
}
