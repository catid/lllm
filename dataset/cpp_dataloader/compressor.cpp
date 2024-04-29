#include "compressor.hpp"

#include "zstd/zstd.h"

#include <cstdlib>
#include <cstdint>

#include <iostream>


//------------------------------------------------------------------------------
// Constants

static const int kCompressionLevel = 1;


//------------------------------------------------------------------------------
// Compressor

bool Compressor::Compress(const void* data, int bytes)
{
    const unsigned max_output_bytes = (unsigned)ZSTD_compressBound(bytes);

    // Space for output
    Result.resize(ZDATA_HEADER_BYTES + max_output_bytes);

    const size_t result = ZSTD_compress(
        Result.data() + ZDATA_HEADER_BYTES,
        max_output_bytes,
        data,
        bytes,
        kCompressionLevel);

    if (ZSTD_isError(result)) {
        return false;
    }

    uint32_t* words = reinterpret_cast<uint32_t*>( Result.data() );
    words[0] = ZDATA_HEADER_MAGIC;
    words[1] = bytes;

    Result.resize(ZDATA_HEADER_BYTES + result);
    return true;
}


//------------------------------------------------------------------------------
// Decompressor

bool Decompressor::Decompress(const void* data, int bytes)
{
    if (bytes < ZDATA_HEADER_BYTES) {
        return false;
    }

    const uint32_t* words = reinterpret_cast<const uint32_t*>( data );
    if (words[0] != ZDATA_HEADER_MAGIC) {
        return false;
    }

    const uint32_t original_size = words[1];
    Result.resize(original_size);

    const size_t result = ZSTD_decompress(
        Result.data(),
        original_size,
        (const uint8_t*)data + ZDATA_HEADER_BYTES,
        bytes - ZDATA_HEADER_BYTES);

    if (ZSTD_isError(result) || result != original_size) {
        return false;
    }

    return true;
}
