#include <mapped_file.hpp>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <iostream>
#include <string>
#include <cstring>

// Test case for opening and reading a mapped file
void TestOpenAndReadFile() {
    const std::string file_name = "test_file.txt";
    const std::string content = "Hello, World!";

    // Create a file and write content to it
    int fd = open(file_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        std::cout << "TestOpenAndReadFile: Failed to create file" << std::endl;
        return;
    }

    if (write(fd, content.data(), content.size()) != static_cast<ssize_t>(content.size())) {
        std::cout << "TestOpenAndReadFile: Failed to write content to file" << std::endl;
        close(fd);
        return;
    }

    close(fd);

    // Open the mapped file using MappedFileReader
    MappedFileReader reader;
    bool open_result = reader.Open(file_name);

    if (!open_result) {
        std::cout << "TestOpenAndReadFile: Failed to open mapped file" << std::endl;
        return;
    }

    // Check if the content of the mapped file matches the original content
    if (reader.GetSize() != content.size() || std::memcmp(reader.GetData(), content.data(), content.size()) != 0) {
        std::cout << "TestOpenAndReadFile: Content of the mapped file does not match the original" << std::endl;
        reader.Close();
        return;
    }

    reader.Close();

    // Clean up the file
    unlink(file_name.c_str());

    std::cout << "TestOpenAndReadFile: Passed" << std::endl;
}

int main() {
    TestOpenAndReadFile();

    return 0;
}
