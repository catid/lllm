#include "compressor.hpp"

#include "tools.hpp"

#include <zstd.h>

#include <cstdlib>
#include <cstdint>


//------------------------------------------------------------------------------
// Constants

static const int kCompressionLevel = 1;
static const int kMaxBytes = 0x00ffffff;


//------------------------------------------------------------------------------
// Compressor

bool Compressor::Compress(const void* data, int bytes, int byte_stride)
{
    if (bytes > kMaxBytes || bytes < 0) {
        LOG_ERROR() << "Compressor: Data too large: bytes=" << bytes;
        return false;
    }

    const void* uncompressed = data;

    if (byte_stride > 1) {
        if (bytes % byte_stride != 0) {
            LOG_ERROR() << "Compressor: Data=" << bytes << " not divisible by byte_stride=" << byte_stride;
            return false;
        }

        Packed.resize(bytes);
        const uint8_t* unpacked = reinterpret_cast<const uint8_t*>(data);
        uint8_t* packed = Packed.data();
        const int plane_bytes = bytes / byte_stride;

        for (int i = 0; i < plane_bytes; ++i) {
            for (int j = 0; j < byte_stride; j++) {
                packed[i + j * plane_bytes] = unpacked[i * byte_stride + j];
            }
        }

        uncompressed = packed;
    }

    // Space for compressed output
    const unsigned max_output_bytes = (unsigned)ZSTD_compressBound(bytes);
    Result.resize(ZDATA_HEADER_BYTES + max_output_bytes);

    const size_t compressed_bytes = ZSTD_compress(
        Result.data() + ZDATA_HEADER_BYTES,
        max_output_bytes,
        uncompressed,
        bytes,
        kCompressionLevel);

    if (ZSTD_isError(compressed_bytes)) {
        LOG_ERROR() << "Compressor: Failed to compress: r=" << compressed_bytes << " err=" << ZSTD_getErrorName(compressed_bytes);
        return false;
    }

    uint8_t* header = Result.data();
    header[0] = static_cast<uint8_t>( bytes );
    header[1] = static_cast<uint8_t>( bytes >> 8 );
    header[2] = static_cast<uint8_t>( bytes >> 16 );

    Result.resize(ZDATA_HEADER_BYTES + compressed_bytes);
    return true;
}


//------------------------------------------------------------------------------
// Decompressor

bool Decompressor::Decompress(const void* data, int bytes, int byte_stride)
{
    if (bytes < ZDATA_HEADER_BYTES) {
        LOG_ERROR() << "Decompressor: Data too small: bytes=" << bytes;
        return false;
    }

    const uint8_t* header = reinterpret_cast<const uint8_t*>( data );
    const uint32_t original_bytes = (uint32_t)header[0] | ((uint32_t)header[1] << 8) | ((uint32_t)header[2] << 16);
    Result.resize(original_bytes);

    uint8_t* decompressed;
    if (byte_stride > 1) {
        Packed.resize(original_bytes);
        decompressed = Packed.data();
    } else {
        decompressed = Result.data();
    }

    const size_t r = ZSTD_decompress(
        decompressed,
        original_bytes,
        header + ZDATA_HEADER_BYTES,
        bytes - ZDATA_HEADER_BYTES);

    if (ZSTD_isError(r) || r != original_bytes) {
        LOG_ERROR() << "Decompressor: Failed to decompress: r=" << r << " original_bytes=" << original_bytes << " err=" << ZSTD_getErrorName(r);
        return false;
    }

    if (byte_stride > 1) {
        if (original_bytes % byte_stride != 0) {
            LOG_ERROR() << "Decompressor: Decompressed data=" << original_bytes << " not divisible by byte_stride=" << byte_stride;
            return false;
        }

        const int plane_bytes = original_bytes / byte_stride;
        const uint8_t* packed = decompressed;
        uint8_t* unpacked = Result.data();

        for (int i = 0; i < plane_bytes; ++i) {
            for (int j = 0; j < byte_stride; j++) {
                unpacked[i * byte_stride + j] = packed[i + j * plane_bytes];
            }
        }
    }

    return true;
}