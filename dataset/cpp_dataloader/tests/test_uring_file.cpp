#include "uring_file.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>

const char* kTestFile = "test_file.bin";
const size_t kTestFileSize = 8192;

void create_test_file() {
    std::cout << "Creating test file..." << std::endl;
    FILE* file = fopen(kTestFile, "wb");
    if (file == nullptr) {
        std::cerr << "Failed to create test file" << std::endl;
        exit(-1);
    }
    for (size_t i = 0; i < kTestFileSize; ++i) {
        char byte = static_cast<char>(rand() % 256);
        fwrite(&byte, 1, 1, file);
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

void read_callback(ssize_t res, void* user_data) {
    std::cout << "Read completed. Bytes read: " << res << std::endl;
    if (res != kTestFileSize / 2) {
        std::cerr << "Read size mismatch. Expected: " << kTestFileSize / 2 << ", Actual: " << res << std::endl;
        exit(-1);
    }
    char* buffer = static_cast<char*>(user_data);
    for (size_t i = 0; i < kTestFileSize / 2; ++i) {
        if (buffer[i] != static_cast<char>(rand() % 256)) {
            std::cerr << "Read data mismatch at index: " << i << std::endl;
            exit(-1);
        }
    }
    free(buffer);
    std::cout << "Read data verified successfully" << std::endl;
}

int main() {
    create_test_file();

    std::cout << "Creating AsyncDiskReader..." << std::endl;
    AsyncDiskReader reader(kTestFile, 1);

    std::cout << "Submitting read request 1..." << std::endl;
    char* buffer1 = static_cast<char*>(malloc(kTestFileSize / 2));
    if (buffer1 == nullptr) {
        std::cerr << "Failed to allocate buffer1" << std::endl;
        remove_test_file();
        return -1;
    }
    reader.submit_read(0, kTestFileSize / 2, read_callback, buffer1);

    std::cout << "Submitting read request 2..." << std::endl;
    char* buffer2 = static_cast<char*>(malloc(kTestFileSize / 2));
    if (buffer2 == nullptr) {
        std::cerr << "Failed to allocate buffer2" << std::endl;
        free(buffer1);
        remove_test_file();
        return -1;
    }
    reader.submit_read(kTestFileSize / 2, kTestFileSize / 2, read_callback, buffer2);

    std::cout << "Waiting for read completions..." << std::endl;
    reader.wait_for_completions();

    remove_test_file();

    std::cout << "All tests passed" << std::endl;
    return 0;
}
