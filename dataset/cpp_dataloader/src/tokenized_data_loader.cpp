#include "tokenized_data_loader.hpp"

#include <algorithm>
#include <random>
#include <iostream>

#include <cpppath.h>
#include <yaml.hpp>


//------------------------------------------------------------------------------
// TokenizedDataLoader

bool TokenizedDataLoader::LoadTokenArrays(const std::string& index_file) {
    MappedFileReader index_reader;
    if (!index_reader.Open(index_file)) {
        return false;
    }

    const uint8_t* index_data = index_reader.GetData();
    size_t index_size = index_reader.GetSize();

    size_t num_data_files = index_size / sizeof(uint64_t);
    const uint64_t* data_file_offsets = reinterpret_cast<const uint64_t*>(index_data);

    data_file_offsets_.assign(data_file_offsets, data_file_offsets + num_data_files);

    std::string data_file_prefix = index_file.substr(0, index_file.find_last_of('.')) + "_";

    for (size_t i = 0; i < num_data_files; ++i) {
        std::string data_file_name = data_file_prefix + std::to_string(i) + ".bin";
        MappedFileReader data_reader;
        if (!data_reader.Open(data_file_name)) {
            return false;
        }
        data_files_.push_back(std::move(data_reader));
    }

    return true;
}

uint64_t TokenizedDataLoader::StartEpoch(uint64_t seed0, uint64_t seed1, uint32_t micro_batch_size, uint32_t context_size) {
    micro_batch_size_ = micro_batch_size;
    context_size_ = context_size;

    uint64_t num_microbatches = data_file_offsets_.back() / (micro_batch_size * context_size * sizeof(uint16_t));
    microbatch_indices_.resize(num_microbatches);
    std::iota(microbatch_indices_.begin(), microbatch_indices_.end(), 0);

    std::seed_seq seed{seed0, seed1};
    std::mt19937 rng(seed);
    std::shuffle(microbatch_indices_.begin(), microbatch_indices_.end(), rng);

    return num_microbatches;
}

bool TokenizedDataLoader::GetTokenArray(
    uint64_t microbatch_index,
    uint32_t context_size,
    uint32_t* micro_batch_size,
    uint32_t* num_tokens,
    uint16_t* output_array)
{
    if (microbatch_index >= microbatch_indices_.size()) {
        return false;
    }

    uint64_t shuffled_index = microbatch_indices_[microbatch_index];
    uint64_t offset = shuffled_index * micro_batch_size_ * context_size_ * sizeof(uint16_t);

    auto it = std::upper_bound(data_file_offsets_.begin(), data_file_offsets_.end(), offset);
    size_t file_index = std::distance(data_file_offsets_.begin(), it) - 1;

    uint64_t file_offset = offset - data_file_offsets_[file_index];
    const uint8_t* compressed_data = data_files_[file_index].GetData() + file_offset;

    Decompressor decompressor;
    if (!decompressor.Decompress(compressed_data, data_files_[file_index].GetSize() - file_offset)) {
        return false;
    }

    uint32_t decompressed_size = decompressor.Result.size();
    uint32_t num_elements = decompressed_size / sizeof(uint16_t);

    std::copy(reinterpret_cast<const uint16_t*>(decompressor.Result.data()),
              reinterpret_cast<const uint16_t*>(decompressor.Result.data()) + num_elements,
              output_array);

    *micro_batch_size = micro_batch_size_;
    *num_tokens = num_elements;

    return true;
}


//------------------------------------------------------------------------------
// Verify

bool data_verify(const char* data_folder_path)
{
    std::string index_file_path = cpppath::join({data_folder_path, DATALOADER_MAIN_INDEX_FILE});

    Yaml::Node root;

    try {
        Yaml::Parse(root, index_file_path);

        Yaml::Node data_items = root["data_files"];

        for(auto it = data_items.Begin(); it != data_items.End(); it++)
        {
            std::cout << (*it).first << ": " << (*it).second.As<string>() << std::endl;
        }

    } catch (Yaml::Exception& e) {
        std::cerr << "Error parsing index file: " << e.what() << std::endl;
        return false;
    }

    WorkerPool pool;
    pool.Start();

    const int num_workers = pool.GetWorkerCount();

    for (int i = 0; i < num_workers; ++i) {
        pool.QueueTask([&]() {
            // FIXME
        });
    }
}