#include "tokenized_data_prep.hpp"

#include <city.h>
#include <cpppath.h>

#include <iostream>


//------------------------------------------------------------------------------
// CompressorContext

bool CompressorContext::WriteTokenizedText(
    const uint32_t* tokenized_text,
    uint32_t text_length,
    OnFileStart on_file_start)
{
    bool r = compressor.Compress(
        tokenized_text,
        text_length * sizeof(uint32_t),
        4);
    if (!r) {
        std::cerr << "Compression failed" << std::endl;
        return false;
    }

    if (current_file_bytes_ == 0) {
        on_file_start(data_file_path_, index_file_path_);

        current_index_.open(index_file_path_, std::ios::binary);
        current_file_.open(data_file_path_, std::ios::binary);
        if (!current_index_ || !current_file_) {
            std::cerr << "Failed to open files: " << index_file_path_ << ", " << data_file_path_ << std::endl;
            return false;
        }
    }

    // Write the offset of the current chunk to the index file
    uint32_t current_offset = static_cast<uint32_t>(current_file_bytes_);
    current_index_.write(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));
    current_index_hash_ ^= CityHash64(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));

    // Write the compressed data to the current file
    current_file_.write(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());
    current_file_bytes_ += compressor.Result.size();
    current_file_hash_ ^= CityHash64(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());

    if (current_file_.fail() || current_index_.fail()) {
        std::cerr << "Failed to write to files" << std::endl;
        return false;
    }

    if (current_file_bytes_ >= kMaxFileSize) {
        // Last 64 bits is the file hash for verification
        current_file_.write(reinterpret_cast<const char*>(&current_file_hash_), sizeof(current_file_hash_));
        current_index_.write(reinterpret_cast<const char*>(&current_index_hash_), sizeof(current_index_hash_));

        if (current_file_.fail() || current_index_.fail()) {
            std::cerr << "Failed to write tail on files" << std::endl;
            return false;
        }

        current_file_.close();
        current_index_.close();
        current_file_hash_ = 0;
        current_index_hash_ = 0;
        current_file_bytes_ = 0;
    }

    return true;
}


//------------------------------------------------------------------------------
// TokenizedDataPrep

void TokenizedDataPrep::Start(const std::string& data_folder_path)
{
    data_folder_path_ = data_folder_path;

    current_file_number_ = 0;
    worker_error_ = false;

    pool_.Start();
    contexts_.resize(pool_.GetWorkerCount());
    for (int i = 0; i < (int)contexts_.size(); ++i) {
        contexts_[i] = std::make_shared<CompressorContext>();
    }
}

bool TokenizedDataPrep::WriteTokenizedText(
    const uint32_t* tokenized_text,
    uint32_t text_length)
{
    // Do not allow workers to run ahead more than N tasks at a time
    const int max_active_tasks = 4;

    OnFileStart on_file_start = [this](std::string& data_file_path, std::string& index_file_path) {
        std::lock_guard<std::mutex> lock(global_index_mutex_);

        std::string data_file_name = "data_" + std::to_string(current_file_number_) + ".bin";
        data_file_path = cpppath::join({data_folder_path_, data_file_name});
        global_index_["data_files"].PushBack() = data_file_path;

        std::string index_file_name = "index_" + std::to_string(current_file_number_) + ".bin";
        index_file_path = cpppath::join({data_folder_path_, index_file_name});
        global_index_["index_files"].PushBack() = index_file_path;

        current_file_number_++;
    };

    std::shared_ptr<TokenizedBuffer> buffer = allocator_.Allocate(tokenized_text, text_length);
    pool_.QueueTask([this, on_file_start, buffer](int worker_index) {
        bool r = contexts_[worker_index]->WriteTokenizedText(
            buffer->text.data(),
            buffer->text.size(),
            on_file_start);
        if (!r) {
            std::cerr << "Worker encountered an error" << std::endl;
            worker_error_ = true;
        }
        allocator_.Free(buffer);
    }, max_active_tasks);

    return !worker_error_;
}

bool TokenizedDataPrep::Finalize() {
    // Wait for all tasks to complete before finalizing the index file
    pool_.WaitForTasks();

    std::string index_file_name = "index.yaml";
    std::string index_file_path = cpppath::join({data_folder_path_, index_file_name});

    try {
        Yaml::Serialize(global_index_, index_file_path);
    } catch (const Yaml::Exception& e) {
        std::cerr << "Failed to serialize index file: " << e.what() << std::endl;
        return false;
    }

    return true;
}
