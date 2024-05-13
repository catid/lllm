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
        kCompressorByteStride);
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
    char current_offset_buffer[4];
    write_uint32_le(current_offset_buffer, current_file_bytes_);
    current_index_.write(current_offset_buffer, sizeof(current_offset_buffer));
    current_index_hash_ ^= CityHash64(current_offset_buffer, sizeof(current_offset_buffer));

    // Write the compressed data to the current file
    current_file_.write(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());
    current_file_bytes_ += compressor.Result.size();
    current_file_hash_ ^= CityHash64(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());

    if (current_file_.fail() || current_index_.fail()) {
        std::cerr << "Failed to write to files" << std::endl;
        return false;
    }

    if (current_file_bytes_ >= kMaxFileSize) {
        if (!FinishCurrentFile()) {
            return false;
        }
    }

    return true;
}

bool CompressorContext::FinishCurrentFile()
{
    /*
        End of index file format:
            <final data file size (4 bytes)>
            <max region size (4 bytes)>
            <final index file hash (8 bytes)>

        The hash also includes the final two fields, hashed individually for consistency.
    */
    char current_offset_buffer[4];
    write_uint32_le(current_offset_buffer, current_file_bytes_);
    current_index_.write(current_offset_buffer, sizeof(current_offset_buffer));
    current_index_hash_ ^= CityHash64(current_offset_buffer, sizeof(current_offset_buffer));

    /*
        End of data file format:
            <final data file hash (8 bytes)>
    */

    char file_hash_buffer[8], index_hash_buffer[8];
    write_uint64_le(file_hash_buffer, current_file_hash_);
    write_uint64_le(index_hash_buffer, current_index_hash_);
    current_file_.write(file_hash_buffer, sizeof(file_hash_buffer));
    current_index_.write(index_hash_buffer, sizeof(index_hash_buffer));

    if (current_file_.fail() || current_index_.fail()) {
        std::cerr << "Failed to write tail on files" << std::endl;
        return false;
    }

    current_file_.close();
    current_index_.close();
    current_file_hash_ = 0;
    current_index_hash_ = 0;
    current_file_bytes_ = 0;
    return true;
}


//------------------------------------------------------------------------------
// TokenizedDataPrep

void TokenizedDataPrep::Start(const std::string& data_folder_path)
{
    data_folder_path_ = data_folder_path;

    current_file_number_ = 0;
    worker_error_ = false;
    stopped_ = false;

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
        data_files_.push_back(data_file_name);

        std::string index_file_name = "index_" + std::to_string(current_file_number_) + ".bin";
        index_file_path = cpppath::join({data_folder_path_, index_file_name});
        index_files_.push_back(index_file_name);

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

bool TokenizedDataPrep::Stop() {
    if (stopped_) {
        return true;
    }
    stopped_ = true;

    // Wait for all tasks to complete before finalizing the index file
    pool_.WaitForTasks();

    for (int i = 0; i < (int)contexts_.size(); ++i) {
        if (contexts_[i] && !contexts_[i]->FinishCurrentFile()) {
            return false;
        }
    }

    std::string index_file_name = "index.yaml";
    std::string index_file_path = cpppath::join({data_folder_path_, index_file_name});

    std::ofstream global_index_file(index_file_path);
    if (!global_index_file.is_open()) {
        std::cerr << "Failed to open index file: " << index_file_path << std::endl;
        return false;
    }

    global_index_file << "data_files:" << std::endl;
    for(auto it = data_files_.begin(); it != data_files_.end(); it++) {
        global_index_file << "  - " << *it << std::endl;
    }

    global_index_file << "index_files:" << std::endl;
    for(auto it = index_files_.begin(); it != index_files_.end(); it++) {
        global_index_file << "  - " << *it << std::endl;
    }

    return true;
}
