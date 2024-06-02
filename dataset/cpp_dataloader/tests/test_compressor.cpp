#include <compressor.hpp>

#include "tools.hpp"

#include <cstdlib>
#include <cstdint>
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
bool TestCompressDecompressString() {
    Compressor compressor;
    Decompressor decompressor;

    std::string original = "Hello, World!";

    // Compress the string
    bool compress_result = compressor.Compress(original.data(), original.size());

    if (!compress_result) {
        LOG_ERROR() << "TestCompressDecompressString: Compression failed";
        return false;
    }

    // Get the compressed data
    const std::vector<uint8_t>& compressed_data = compressor.Result;

    // Decompress the compressed data
    bool decompress_result = decompressor.Decompress(compressed_data.data(), compressed_data.size());

    if (!decompress_result) {
        LOG_ERROR() << "TestCompressDecompressString: Decompression failed";
        return false;
    }

    // Get the decompressed data
    const std::vector<uint8_t>& decompressed_data = decompressor.Result;

    // Convert decompressed data to string
    std::string decompressed_string(decompressed_data.begin(), decompressed_data.end());

    // Check if the decompressed string matches the original string
    if (original != decompressed_string) {
        LOG_ERROR() << "TestCompressDecompressString: Decompressed string does not match the original";
        return false;
    }

    LOG_INFO() << "TestCompressDecompressString: Passed";
    return true;
}

// Test case for compressing and decompressing a vector of integers
bool TestCompressDecompressVector() {
    Compressor compressor;
    Decompressor decompressor;

    std::vector<int> original = {1, 2, 3, 4, 5};

    // Compress the vector
    bool compress_result = compressor.Compress(original.data(), original.size() * sizeof(int));

    if (!compress_result) {
        LOG_ERROR() << "TestCompressDecompressVector: Compression failed";
        return false;
    }

    // Get the compressed data
    const std::vector<uint8_t>& compressed_data = compressor.Result;

    // Decompress the compressed data
    bool decompress_result = decompressor.Decompress(compressed_data.data(), compressed_data.size());

    if (!decompress_result) {
        LOG_ERROR() << "TestCompressDecompressVector: Decompression failed";
        return false;
    }

    // Get the decompressed data
    const std::vector<uint8_t>& decompressed_data = decompressor.Result;

    // Convert decompressed data to vector of integers
    std::vector<int> decompressed_vector(decompressed_data.size() / sizeof(int));
    memcpy(decompressed_vector.data(), decompressed_data.data(), decompressed_data.size());

    // Check if the decompressed vector matches the original vector
    if (!AreVectorsEqual(original, decompressed_vector)) {
        LOG_ERROR() << "TestCompressDecompressVector: Decompressed vector does not match the original";
        return false;
    }

    LOG_INFO() << "TestCompressDecompressVector: Passed";
    return true;
}

// Test case for compressing and decompressing with different byte strides
bool TestCompressDecompressByteStride() {
    Compressor compressor;
    Decompressor decompressor;

    // Generate test data
    const int data_size = 1024 * 3 * 5 * 7;
    std::vector<uint8_t> original_data(data_size);
    for (int i = 0; i < data_size; ++i) {
        original_data[i] = static_cast<uint8_t>(i % 256);
    }

    // Test byte strides from 1 to 8
    for (int byte_stride = 1; byte_stride <= 8; ++byte_stride) {
        // Compress the data with the current byte stride
        bool compress_result = compressor.Compress(original_data.data(), original_data.size(), byte_stride);
        if (!compress_result) {
            LOG_ERROR() << "TestCompressDecompressByteStride: Compression failed for byte_stride " << byte_stride;
            return false;
        }

        // Get the compressed data
        const std::vector<uint8_t>& compressed_data = compressor.Result;

        // Decompress the compressed data with the same byte stride
        bool decompress_result = decompressor.Decompress(compressed_data.data(), compressed_data.size(), byte_stride);
        if (!decompress_result) {
            LOG_ERROR() << "TestCompressDecompressByteStride: Decompression failed for byte_stride " << byte_stride;
            return false;
        }

        // Get the decompressed data
        const std::vector<uint8_t>& decompressed_data = decompressor.Result;

        // Check if the decompressed data matches the original data
        if (!AreVectorsEqual(original_data, decompressed_data)) {
            LOG_ERROR() << "TestCompressDecompressByteStride: Decompressed data does not match the original for byte_stride " << byte_stride;
            return false;
        }
    }

    LOG_INFO() << "TestCompressDecompressByteStride: Passed";
    return true;
}

int main() {
    if (!TestCompressDecompressString()) {
        return -1;
    }

    if (!TestCompressDecompressVector()) {
        return -1;
    }

    if (!TestCompressDecompressByteStride()) {
        return -1;
    }

    LOG_INFO() << "All tests passed";
    return 0;
}
