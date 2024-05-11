#include "tokenized_data_loader.hpp"
#include "tools.hpp"

#include <algorithm>
#include <random>
#include <iostream>

#include <cpppath.h>
#include <ryml.hpp>
#include <mapped_file.hpp>
#include <city.h>


//------------------------------------------------------------------------------
// GlobalIndexYaml

bool GlobalIndexYaml::Read(const std::string& data_folder_path) {
    std::string index_file_path = cpppath::join({data_folder_path, DATALOADER_MAIN_INDEX_FILE});

    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        return false;
    }

    ryml::csubstr yaml_substr((const char*)index_reader.GetData(), index_reader.GetSize());
    ryml::Tree tree = ryml::parse(yaml_substr);
    ryml::ConstNodeRef data_files = tree["data_files"];
    ryml::ConstNodeRef index_files = tree["index_files"];
    if (data_files.invalid() || index_files.invalid() ||
        !data_files.is_seq() || !index_files.is_seq() ||
        data_files.num_children() != index_files.num_children()) {
        return false;
    }

    const int num_files = (int)data_files.num_children();
    for (int i = 0; i < num_files; ++i) {
        std::string data_file, index_file;
        ryml::from_chars(data_files[i].val(), &data_file);
        ryml::from_chars(index_files[i].val(), &index_file);

        std::string data_file_path = cpppath::join({data_folder_path, data_file});
        std::string index_file_path = cpppath::join({data_folder_path, index_file});

        data_files_.push_back(data_file_path);
        index_files_.push_back(index_file_path);
    }

    return true;
}


//------------------------------------------------------------------------------
// TokenizedDataLoader

bool TokenizedDataLoader::Start(const std::string& data_folder_path) {
    if (!global_index_yaml_.Read(data_folder_path)) {
        std::cout << "Failed to read global index file at " << data_folder_path << std::endl;
        return false;
    }

    uint64_t total_num_regions = 0;

    const int num_files = (int)global_index_yaml_.data_files_.size();
    for (int i = 0; i < num_files; ++i) {
        const std::string& data_file_path = global_index_yaml_.data_files_[i];
        const std::string& index_file_path = global_index_yaml_.index_files_[i];

        std::shared_ptr<MappedFileReader> data_reader = std::make_shared<MappedFileReader>();
        if (!data_reader->Open(data_file_path)) {
            std::cout << "Failed to open data file at " << data_file_path << std::endl;
            return false;
        }

        std::shared_ptr<MappedFileReader> index_reader = std::make_shared<MappedFileReader>();
        if (!index_reader->Open(index_file_path)) {
            std::cout << "Failed to open index file at " << index_file_path << std::endl;
            return false;
        }

        uint64_t index_file_bytes = index_reader->GetSize();
        uint64_t num_regions = (index_file_bytes - 8 - 2) / sizeof(uint32_t);
        if (num_regions > UINT32_MAX) {
            std::cout << "Too many regions in index file at " << index_file_path << std::endl;
            return false;
        }
        num_regions_.push_back(num_regions);
        total_num_regions += num_regions;
    };

    total_num_regions_ = total_num_regions;
    if (total_num_regions_ >= UINT32_MAX) {
        std::cout << "FIXME: Too many regions overall: " << total_num_regions_ << std::endl;
        return false;
    }

    pool_.Start();

    return true;
}

void TokenizedDataLoader::Stop()
{
    pool_.Stop();
}

uint64_t TokenizedDataLoader::StartEpoch(uint64_t seed0, uint64_t seed1, uint32_t micro_batch_size, uint32_t context_size) {
    micro_batch_size_ = micro_batch_size;
    context_size_ = context_size;

    microbatch_indices_.resize(total_num_regions_);
    for (uint64_t i = 0; i < total_num_regions_; ++i) {
        microbatch_indices_[i] = i;
    }

    // Randomly order the microbatches
    std::seed_seq seed{seed0, seed1};
    std::mt19937 rng(seed);
    std::shuffle(microbatch_indices_.begin(), microbatch_indices_.end(), rng);

    for (uint32_t i = 0; i < micro_batch_size * 2; ++i) {
        pool_.QueueTask([this](int worker_index) {
            
        },
    }

    return total_num_regions_;
}

bool TokenizedDataLoader::GetTokenArray(
    uint64_t microbatch_index,
    uint32_t context_size,
    uint32_t* micro_batch_size,
    uint32_t* num_tokens,
    uint32_t* output_array)
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

bool verify_files(
    const std::string& index_file_path,
    const std::string& data_file_path)
{
    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        std::cout << "Failed to open index file: " << index_file_path << std::endl;
        return false;
    }

    const char* index_data = reinterpret_cast<const char*>(index_reader.GetData());
    size_t index_size = index_reader.GetSize();
    uint64_t index_hash = read_uint64_le(index_data + index_size - 8);
    index_hash ^= CityHash64(index_data, sizeof(uint32_t));
    size_t word_count = (index_size - 8) / sizeof(uint32_t);

    MappedFileReader data_reader;
    if (!data_reader.Open(data_file_path)) {
        std::cout << "Failed to open data file: " << data_file_path << std::endl;
        return false;
    }

    const char* data_data = reinterpret_cast<const char*>(data_reader.GetData());
    size_t data_size = data_reader.GetSize();
    uint64_t data_hash = read_uint64_le(data_data + data_size - 8);

    for (size_t i = 1; i < word_count; ++i) {
        const char* current_offset_buffer = index_data + i * sizeof(uint32_t);
        uint32_t start = read_uint32_le(current_offset_buffer - 4);
        uint32_t end = read_uint32_le(current_offset_buffer);
        if (end <= start) {
            std::cout << "Invalid offset: " << current_offset_buffer << std::endl;
            return false;
        }
        uint32_t bytes = end - start;

        index_hash ^= CityHash64(current_offset_buffer, sizeof(uint32_t));

        data_hash ^= CityHash64(data_data + start, bytes);
    }

    if (index_hash != 0) {
        std::cout << "Index file is corrupted: " << index_file_path << std::endl;
        return false;
    }

    if (data_hash != 0) {
        std::cout << "Data file is corrupted: " << data_file_path << std::endl;
        return false;
    }

    return true;
}

bool data_verify(const std::string& data_folder_path)
{
    GlobalIndexYaml global_index_yaml;
    if (!global_index_yaml.Read(data_folder_path)) {
        std::cout << "Failed to read global index file at " << data_folder_path << std::endl;
        return false;
    }

    WorkerPool pool;
    pool.Start();

    std::atomic<bool> data_error(false);
    std::atomic<int> files_verified(0);

    const int num_files = (int)global_index_yaml.data_files_.size();
    for (int i = 0; i < num_files; ++i) {
        const std::string& data_file_path = global_index_yaml.data_files_[i];
        const std::string& index_file_path = global_index_yaml.index_files_[i];

        const int max_active_tasks = 2;
        pool.QueueTask([&files_verified, num_files, &data_error, data_file_path, index_file_path](int worker_index) {
            if (data_error) {
                return;
            }
            if (!verify_files(index_file_path, data_file_path)) {
                data_error = true;
            }
            const int count = files_verified++;
            if (count % 10 == 0) {
                std::cout << "verified " << count << "/" << num_files << std::endl;
            }
        }, max_active_tasks);

        if (data_error) {
            break;
        }
    }

    pool.WaitForTasks();
    pool.Stop();

    if (data_error) {
        std::cout << "Data verification failed" << std::endl;
        return false;
    }

    return true;
}
