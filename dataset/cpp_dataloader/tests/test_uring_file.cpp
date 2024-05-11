#include <iostream>
#include <cstdlib>
#include <cstring>
#include "async_disk_reader.h"

const char* kTestFile = "test_file.bin";
const size_t kTestFileSize = 8192;

void create_test_file() {
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
}

void remove_test_file() {
    remove(kTestFile);
}

void read_callback(ssize_t res, void* user_data) {
    if (res != kTestFileSize / 2) {
        std::cerr << "Read size mismatch" << std::endl;
        exit(-1);
    }
    char* buffer = static_cast<char*>(user_data);
    for (size_t i = 0; i < kTestFileSize / 2; ++i) {
        if (buffer[i] != static_cast<char>(rand() % 256)) {
            std::cerr << "Read data mismatch" << std::endl;
            exit(-1);
        }
    }
    free(buffer);
}

int main() {
    create_test_file();

    AsyncDiskReader reader(kTestFile, 1);

    char* buffer1 = static_cast<char*>(malloc(kTestFileSize / 2));
    if (buffer1 == nullptr) {
        std::cerr << "Failed to allocate buffer1" << std::endl;
        remove_test_file();
        return -1;
    }
    reader.submit_read(0, kTestFileSize / 2, read_callback, buffer1);

    char* buffer2 = static_cast<char*>(malloc(kTestFileSize / 2));
    if (buffer2 == nullptr) {
        std::cerr << "Failed to allocate buffer2" << std::endl;
        free(buffer1);
        remove_test_file();
        return -1;
    }
    reader.submit_read(kTestFileSize / 2, kTestFileSize / 2, read_callback, buffer2);

    reader.wait_for_completions();

    remove_test_file();

    std::cout << "All tests passed" << std::endl;
    return 0;
}
