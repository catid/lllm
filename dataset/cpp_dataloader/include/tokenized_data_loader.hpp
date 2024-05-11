#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dataloader.hpp"
#include "compressor.hpp"
#include "mapped_file.hpp"
#include "worker_pool.hpp"


//------------------------------------------------------------------------------
// GlobalIndexYaml

struct GlobalIndexYaml {
    bool Read(const std::string& data_folder_path);

    std::vector<std::string> data_files_, index_files_;
};


//------------------------------------------------------------------------------
// TokenizedDataLoader

class TokenizedDataLoader {
public:
    ~TokenizedDataLoader() {
        Stop();
    }

    bool Start(const std::string& data_folder_path);

    void Stop();

    uint64_t StartEpoch(
        uint64_t seed0,
        uint64_t seed1,
        uint32_t micro_batch_size,
        uint32_t context_size);

    bool GetTokenArray(
        uint64_t microbatch_index,
        uint32_t context_size,
        uint32_t* micro_batch_size,
        uint32_t* num_tokens,
        uint32_t* output_array);

private:
    GlobalIndexYaml global_index_yaml_;
    std::vector<uint64_t> num_regions_;
    uint64_t total_num_regions_ = 0;

    std::vector<std::shared_ptr<MappedFileReader>> index_files_;

    uint32_t micro_batch_size_ = 0;
    uint32_t context_size_ = 0;

    std::vector<uint32_t> microbatch_indices_;

    WorkerPool pool_;
};


//------------------------------------------------------------------------------
// Verify

bool data_verify(const std::string& data_folder_path);
