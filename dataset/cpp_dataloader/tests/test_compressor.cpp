#include <compressor.hpp>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <string.h>

// Helper function to check if two vectors are equal
template <typename T>
bool AreVectorsEqual(const std::vector<T>& v1, const std::vector<T>& v2) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i] != v2[i]) {
            return false;
        }
    }

    return true;
}

// Test case for compressing and decompressing a string
void TestCompressDecompressString() {
    Compressor compressor;
    Decompressor decompressor;

    std::string original = "Hello, World!";

    // Compress the string
    bool compress_result = compressor.Compress(original.data(), original.size());

    if (!compress_result) {
        std::cout << "TestCompressDecompressString: Compression failed" << std::endl;
        return;
    }

    // Get the compressed data
    const std::vector<uint8_t>& compressed_data = compressor.Result;

    // Decompress the compressed data
    bool decompress_result = decompressor.Decompress(compressed_data.data(), compressed_data.size());

    if (!decompress_result) {
        std::cout << "TestCompressDecompressString: Decompression failed" << std::endl;
        return;
    }

    // Get the decompressed data
    const std::vector<uint8_t>& decompressed_data = decompressor.Result;

    // Convert decompressed data to string
    std::string decompressed_string(decompressed_data.begin(), decompressed_data.end());

    // Check if the decompressed string matches the original string
    if (original != decompressed_string) {
        std::cout << "TestCompressDecompressString: Decompressed string does not match the original" << std::endl;
        return;
    }

    std::cout << "TestCompressDecompressString: Passed" << std::endl;
}

// Test case for compressing and decompressing a vector of integers
void TestCompressDecompressVector() {
    Compressor compressor;
    Decompressor decompressor;

    std::vector<int> original = {1, 2, 3, 4, 5};

    // Compress the vector
    bool compress_result = compressor.Compress(original.data(), original.size() * sizeof(int));

    if (!compress_result) {
        std::cout << "TestCompressDecompressVector: Compression failed" << std::endl;
        return;
    }

    // Get the compressed data
    const std::vector<uint8_t>& compressed_data = compressor.Result;

    // Decompress the compressed data
    bool decompress_result = decompressor.Decompress(compressed_data.data(), compressed_data.size());

    if (!decompress_result) {
        std::cout << "TestCompressDecompressVector: Decompression failed" << std::endl;
        return;
    }

    // Get the decompressed data
    const std::vector<uint8_t>& decompressed_data = decompressor.Result;

    // Convert decompressed data to vector of integers
    std::vector<int> decompressed_vector(decompressed_data.size() / sizeof(int));
    memcpy(decompressed_vector.data(), decompressed_data.data(), decompressed_data.size());

    // Check if the decompressed vector matches the original vector
    if (!AreVectorsEqual(original, decompressed_vector)) {
        std::cout << "TestCompressDecompressVector: Decompressed vector does not match the original" << std::endl;
        return;
    }

    std::cout << "TestCompressDecompressVector: Passed" << std::endl;
}

int main() {
    TestCompressDecompressString();
    TestCompressDecompressVector();

    return 0;
}
