#include "compressor.hpp"

#include "tools.hpp"

#include <zstd.h>

#include <cstdlib>
#include <cstdint>


//------------------------------------------------------------------------------
// Compressor

bool Compressor::Compress(
    const void* token_data,
    uint32_t token_count,
    uint32_t token_bytes)
{
    if (!token_data || token_count <= 0 || token_bytes <= 0) {
        LOG_ERROR() << "Compressor: Invalid parameters token_data=" << token_data
            << " token_count=" << token_count << " token_bytes=" << token_bytes;
        return false;
    }

    const void* uncompressed = token_data;
    const int uncompressed_bytes = token_count * token_bytes;

    if (token_bytes > 1) {
        Packed.resize(uncompressed_bytes);
        const uint8_t* unpacked = reinterpret_cast<const uint8_t*>(token_data);

        uint8_t* packed = Packed.data();

        for (uint32_t i = 0; i < token_count; ++i) {
            for (uint32_t j = 0; j < token_bytes; j++) {
                packed[i + j * token_count] = unpacked[i * token_bytes + j];
            }
        }

        uncompressed = packed;
    }

    // Space for compressed output
    const uint32_t max_output_bytes = (uint32_t)ZSTD_compressBound(uncompressed_bytes);
    Result.resize(max_output_bytes);

    const size_t compressed_bytes = ZSTD_compress(
        Result.data(),
        max_output_bytes,
        uncompressed,
        uncompressed_bytes,
        kZstdCompressionLevel);

    if (ZSTD_isError(compressed_bytes)) {
        LOG_ERROR() << "Compressor: Failed to compress: r=" << compressed_bytes
            << " err=" << ZSTD_getErrorName(compressed_bytes);
        return false;
    }

    Result.resize(compressed_bytes);
    return true;
}


//------------------------------------------------------------------------------
// Decompressor

bool Decompressor::Decompress(
    const void* compressed_data,
    uint32_t compressed_bytes,
    uint32_t original_tokens,
    uint32_t token_bytes)
{
    if (!compressed_data || compressed_bytes <= 0) {
        LOG_ERROR() << "Decompressor: Invalid compressed bytes=" << compressed_bytes;
        return false;
    }
    if (original_tokens <= 0 || token_bytes <= 0) {
        LOG_ERROR() << "Decompressor: Invalid data=" << original_tokens
            << " token_bytes=" << token_bytes;
        return false;
    }

    const uint32_t original_bytes = original_tokens * token_bytes;

    Result.resize(original_tokens);
    Packed.resize(original_bytes);
    uint8_t* decompressed = Packed.data();

    const size_t r = ZSTD_decompress(
        decompressed,
        original_bytes,
        compressed_data,
        compressed_bytes);

    if (ZSTD_isError(r) || r != (size_t)original_bytes) {
        LOG_ERROR() << "Decompressor: Failed to decompress: r=" << r
            << " original_bytes=" << original_bytes << " err=" << ZSTD_getErrorName(r);
        return false;
    }

    uint32_t* unpacked = reinterpret_cast<uint32_t*>( Result.data() );

    for (uint32_t i = 0; i < original_tokens; ++i) {
        uint8_t unpacked_word[8] = {0};
        for (uint32_t j = 0; j < token_bytes; j++) {
            unpacked_word[j] = decompressed[i + j * original_tokens];
        }

        unpacked[i] = read_uint32_le(unpacked_word);
    }

    return true;
}
