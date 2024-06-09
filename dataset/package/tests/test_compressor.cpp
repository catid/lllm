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
    bool decompress_result = decompressor.Decompress(
        compressed_data.data(),
        compressed_data.size(),
        original.size());

    if (!decompress_result) {
        LOG_ERROR() << "TestCompressDecompressString: Decompression failed";
        return false;
    }

    // Get the decompressed data
    const std::vector<int32_t>& decompressed_data = decompressor.Result;

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

    std::vector<uint32_t> original = {1, 2, 3, 4, 56789};

    // Compress the vector
    bool compress_result = compressor.Compress(
        original.data(),
        original.size(),
        4);

    if (!compress_result) {
        LOG_ERROR() << "TestCompressDecompressVector: Compression failed";
        return false;
    }

    // Get the compressed data
    const std::vector<uint8_t>& compressed_data = compressor.Result;

    // Decompress the compressed data
    bool decompress_result = decompressor.Decompress(
        compressed_data.data(),
        compressed_data.size(),
        original.size(),
        4);

    if (!decompress_result) {
        LOG_ERROR() << "TestCompressDecompressVector: Decompression failed";
        return false;
    }

    // Get the decompressed data
    const std::vector<int32_t>& decompressed_data = decompressor.Result;

    // Convert decompressed data to vector of integers
    std::vector<uint32_t> decompressed_vector(decompressed_data.size());
    for (uint32_t i = 0; i < decompressed_data.size(); ++i) {
        decompressed_vector[i] = decompressed_data[i];
    }

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

    // Test token_bytes from 1 to 4
    for (int token_bytes = 1; token_bytes <= 4; ++token_bytes) {
        const int token_count = original_data.size() / token_bytes;

        // Compress the data with the current byte stride
        bool compress_result = compressor.Compress(
            original_data.data(),
            token_count,
            token_bytes);
        if (!compress_result) {
            LOG_ERROR() << "TestCompressDecompressByteStride: Compression failed for token_bytes " << token_bytes;
            return false;
        }

        // Decompress the compressed data with the same byte stride
        bool decompress_result = decompressor.Decompress(
            compressor.Result.data(),
            compressor.Result.size(),
            token_count,
            token_bytes);
        if (!decompress_result) {
            LOG_ERROR() << "TestCompressDecompressByteStride: Decompression failed for token_bytes " << token_bytes;
            return false;
        }

        for (int i = 0; i < token_count; ++i) {
            uint32_t original_word = 0;
            for (int j = 0; j < token_bytes; ++j) {
                original_word |= original_data[i * token_bytes + j] << (j * 8);
            }

            if (original_word != (uint32_t)decompressor.Result[i]) {
                LOG_ERROR() << "TestCompressDecompressByteStride: Mismatch for token_bytes="
                    << token_bytes << " at offset " << i;
                return false;
            }
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
