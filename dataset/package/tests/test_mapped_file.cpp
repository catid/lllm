#include <mapped_file.hpp>

#include "tools.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <string>
#include <cstring>

// Test case for opening and reading a mapped file
bool TestOpenAndReadFile() {
    const std::string file_name = "test_file.txt";
    const std::string content = "Hello, World!";

    // Create a file and write content to it
    int fd = open(file_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        LOG_ERROR() << "TestOpenAndReadFile: Failed to create file";
        return false;
    }

    if (write(fd, content.data(), content.size()) != static_cast<ssize_t>(content.size())) {
        LOG_ERROR() << "TestOpenAndReadFile: Failed to write content to file";
        close(fd);
        return false;
    }

    close(fd);

    // Open the mapped file using MappedFileReader
    MappedFileReader reader;
    bool open_result = reader.Open(file_name);

    if (!open_result) {
        LOG_ERROR() << "TestOpenAndReadFile: Failed to open mapped file";
        return false;
    }

    // Check if the content of the mapped file matches the original content
    if (reader.GetSize() != content.size() || std::memcmp(reader.GetData(), content.data(), content.size()) != 0) {
        LOG_ERROR() << "TestOpenAndReadFile: Content of the mapped file does not match the original";
        reader.Close();
        return false;
    }

    reader.Close();

    // Clean up the file
    unlink(file_name.c_str());

    LOG_INFO() << "TestOpenAndReadFile: Passed";
    return true;
}

int main() {
    if (!TestOpenAndReadFile()) {
        return -1;
    }

    LOG_INFO() << "All tests passed";
    return 0;
}
