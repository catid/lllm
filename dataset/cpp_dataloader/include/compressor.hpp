#pragma once

#include <cstdint>
#include <vector>


//------------------------------------------------------------------------------
// Constants

#define ZDATA_HEADER_BYTES 8
#define ZDATA_HEADER_MAGIC 0xCA7DDEED
/*
    Header size: 8 bytes

    <Magic(4 bytes)> <OriginalSize(4 bytes)>
*/


//------------------------------------------------------------------------------
// Compressor

struct Compressor {
    bool Compress(const void* data, int bytes);

    std::vector<uint8_t> Result;
};


//------------------------------------------------------------------------------
// Decompressor

struct Decompressor {
    bool Decompress(const void* data, int bytes);

    std::vector<uint8_t> Result;
};
