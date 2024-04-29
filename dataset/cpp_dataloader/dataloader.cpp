#include "compressor.hpp"
#include "mapped_file.hpp"

#include <iostream>

#include <cstdint>
#include <fstream>

extern "C" {


bool cpp_write_token_arrays(
    const uint32_t* const* token_arrays,
    const size_t* array_sizes,
    size_t num_arrays,
    const char* output_file)
{
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    for (size_t i = 0; i < num_arrays; i++) {
        std::vector<uint16_t> packed(array_sizes[i]);

        const uint32_t array_size = array_sizes[i];
        const uint32_t* token_array = token_arrays[i];

        for (size_t j = 0; j < array_size; j++) {
            packed[j] = static_cast<uint16_t>( token_array[j] );
        }

        Compressor compressor;
        if (!compressor.Compress(packed.data(), packed.size() * sizeof(uint16_t))) {
            std::cerr << "Failed to compress token array." << std::endl;
            return false;
        }

        auto& result = compressor.Result;
        ofs.write(reinterpret_cast<const char*>(result.data()), result.size());
    }

    ofs.close();

    if (!ofs) {
        std::cerr << "Failed to write token arrays to disk." << std::endl;
        return false;
    }

    // FIXME: Write array sizes

    return true;
}


} // extern "C"
