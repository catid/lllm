#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "compressor.hpp"
#include "mapped_file.hpp"

class TokenizedDataLoader {
public:
    bool LoadTokenArrays(const std::string& index_file);
    uint64_t StartEpoch(uint64_t seed0, uint64_t seed1, uint32_t micro_batch_size, uint32_t context_size);
    bool GetTokenArray(uint64_t microbatch_index, uint32_t context_size, uint32_t* micro_batch_size, uint32_t* num_tokens, uint16_t* output_array);

private:
    std::vector<MappedFileReader> data_files_;
    std::vector<uint64_t> data_file_offsets_;
    std::vector<uint64_t> microbatch_indices_;
    uint32_t micro_batch_size_ = 0;
    uint32_t context_size_ = 0;
};
