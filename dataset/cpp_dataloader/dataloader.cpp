#include <iostream>

#include <cstdint>
#include <fstream>

extern "C" {


bool cpp_write_token_arrays(
    const uint32_t* token_arrays,
    size_t array_size,
    const char* output_file)
{
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return false;
    }

    ofs.write(reinterpret_cast<const char*>(token_arrays), array_size * sizeof(uint64_t));
    ofs.close();

    if (!ofs) {
        std::cerr << "Failed to write token arrays to disk." << std::endl;
        return false;
    }

    return true;
}


} // extern "C"
