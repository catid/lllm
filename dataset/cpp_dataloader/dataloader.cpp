#include <iostream>
#include <zstd.h>

extern "C" {
    int compressData(const char* input, int inputSize, char* output, int outputSize) {
        size_t compressedSize = ZSTD_compress(output, outputSize, input, inputSize, 1);
        return static_cast<int>(compressedSize);
    }
}
