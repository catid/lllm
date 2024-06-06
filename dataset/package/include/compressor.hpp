#pragma once

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// Constants

#define ZDATA_HEADER_BYTES 3
/*
    Header size: 3 bytes

    <OriginalSize(3 bytes)>
*/


//------------------------------------------------------------------------------
// Compressor

struct Compressor {
    bool Compress(const void* data, int bytes, int byte_stride = 0);

    std::vector<uint8_t> Packed;
    std::vector<uint8_t> Result;
};


//------------------------------------------------------------------------------
// Decompressor

struct Decompressor {
    bool Decompress(const void* data, int bytes, int byte_stride = 0);

    std::vector<uint8_t> Packed;
    std::vector<uint8_t> Result;
};
