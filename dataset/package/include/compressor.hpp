/*
    Zstd compressor with byte stride support.

    Byte stride means that the data stored is in words rather than bytes.
    Byte stride is the width of the words.

    To improve compression, we put each byte of each word in its own plane
    in the compressed data.  For example if token_bytes=4, then we store
    all of the low bytes of each word together in the first plane.
    In my experience this improves compression because usually the high
    bytes are all zeroes and can be efficiently compressed with RLE.
    Furthermore the top bytes have fewer states so they will compress
    better than the low bytes.
*/

#pragma once

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// Constants

static const int kZstdCompressionLevel = 1;


//------------------------------------------------------------------------------
// Compressor

struct Compressor {
    bool Compress(
        const void* token_data,
        uint32_t token_count,
        uint32_t token_bytes = 1);

    // Result of Compress()
    std::vector<uint8_t> Result;

    uint32_t GetCompressedBytes() const {
        return static_cast<uint32_t>( Result.size() );
    }

private:
    std::vector<uint8_t> Packed;
};


//------------------------------------------------------------------------------
// Decompressor

// This always decompresses to 32-bit words
struct Decompressor {
    bool Decompress(
        const void* compressed_data,
        uint32_t compressed_bytes,
        uint32_t original_tokens,
        uint32_t token_bytes = 1);

    // Result of Decompress()
    std::vector<int32_t> Result;

private:
    std::vector<uint8_t> Packed;
};
