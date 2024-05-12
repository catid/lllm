#include "uring_file.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>

const char* kTestFile = "test_file.bin";
const size_t kTestFileSize = 8192;

std::hash<std::string> hasher;
size_t hash1, hash2;

void create_test_file() {
    std::cout << "Creating test file..." << std::endl;
    FILE* file = fopen(kTestFile, "wb");
    if (file == nullptr) {
        std::cerr << "Failed to create test file" << std::endl;
        exit(-1);
    }
    
    std::string data1, data2;
    data1.reserve(kTestFileSize / 2);
    data2.reserve(kTestFileSize / 2);
    
    for (size_t i = 0; i < kTestFileSize / 2; ++i) {
        char byte = static_cast<char>(rand() % 256);
        data1.push_back(byte);
    }
    
    for (size_t i = 0; i < kTestFileSize / 2; ++i) {
        char byte = static_cast<char>(rand() % 256);
        data2.push_back(byte);
    }
    
    fwrite(data1.data(), 1, data1.size(), file);
    fwrite(data2.data(), 1, data2.size(), file);
    fclose(file);
    
    hash1 = hasher(data1);
    hash2 = hasher(data2);
    
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
    std::cout << "Read completed. Bytes read: " << res << std::endl;
    
    if (res != kTestFileSize / 2) {
        std::cerr << "Read size mismatch. Expected: " << kTestFileSize / 2 << ", Actual: " << res << std::endl;
        exit(-1);
    }
    
    std::string data(reinterpret_cast<char*>(buffer), res);
    size_t expected_hash = *reinterpret_cast<size_t*>(user_data);
    size_t actual_hash = hasher(data);
    
    if (actual_hash != expected_hash) {
        std::cerr << "Read data mismatch. Expected hash: " << expected_hash << ", Actual hash: " << actual_hash << std::endl;
        exit(-1);
    }
    
    std::cout << "Read data verified successfully" << std::endl;
}

int main() {
    create_test_file();
    
    std::cout << "Creating AsyncUringReader..." << std::endl;
    AsyncUringReader reader;
    if (!reader.Open(kTestFile, 2)) {
        std::cerr << "Failed to open AsyncUringReader" << std::endl;
        remove_test_file();
        return -1;
    }
    
    std::cout << "Submitting read request 1..." << std::endl;
    if (!reader.Read(0, kTestFileSize / 2, read_callback, &hash1)) {
        std::cerr << "Failed to submit read request 1" << std::endl;
        reader.Close();
        remove_test_file();
        return -1;
    }
    
    std::cout << "Submitting read request 2..." << std::endl;
    if (!reader.Read(kTestFileSize / 2, kTestFileSize / 2, read_callback, &hash2)) {
        std::cerr << "Failed to submit read request 2" << std::endl;
        reader.Close();
        remove_test_file();
        return -1;
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