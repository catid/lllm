#include "tokenized_data_prep.hpp"

#include <city.h>
#include <cpppath.h>

//------------------------------------------------------------------------------
// CompressorContext

bool CompressorContext::WriteTokens(
    const void* tokens,
    uint32_t token_count,
    OnFileStart on_file_start)
{
    bool r = compressor_.Compress(
        tokens,
        token_count,
        token_bytes_);
    if (!r) {
        LOG_ERROR() << "Compression failed";
        return false;
    }

    if (current_file_bytes_ == 0) {
        on_file_start(data_file_path_, index_file_path_);

        current_index_.open(index_file_path_, std::ios::binary);
        current_file_.open(data_file_path_, std::ios::binary);
        if (!current_index_ || !current_file_) {
            LOG_ERROR() << "Failed to open files: " << index_file_path_ << ", " << data_file_path_;
            return false;
        }
    }

    // Write the offset of the compressed text, and its original length to the index file
    char index_data[kIndexRecordBytes];
    write_uint32_le(index_data, current_file_bytes_);
    write_uint32_le(index_data + 4, token_count);
    current_index_.write(index_data, kIndexRecordBytes);
    current_index_hash_ ^= CityHash64(index_data, kIndexRecordBytes);

    // Write the compressed data to the current file
    const char* compressed_data = reinterpret_cast<const char*>(compressor_.Result.data());
    const int compressed_bytes = compressor_.Result.size();
    current_file_.write(compressed_data, compressed_bytes);
    current_file_hash_ ^= CityHash64(compressed_data, compressed_bytes);

    current_file_bytes_ += compressed_bytes;

    if (current_file_.fail() || current_index_.fail()) {
        LOG_ERROR() << "Failed to write to files";
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
    if (current_file_bytes_ == 0) {
        return true;
    }

    // Write end of index file
    {
        char index_end[kIndexEndBytes];
        write_uint32_le(index_end, current_file_bytes_);
        index_end[4] = static_cast<uint8_t>( DATALOADER_VERSION );
        index_end[5] = static_cast<uint8_t>( token_bytes_ );
        current_index_hash_ ^= CityHash64(index_end, kIndexEndBytes - 8);
        write_uint64_le(index_end + kIndexEndBytes - 8, current_index_hash_);
        current_index_.write(index_end, kIndexEndBytes);
    }

    // Write end of data file
    {
        char data_end[kDataEndBytes];
        data_end[0] = static_cast<uint8_t>( DATALOADER_VERSION );
        data_end[1] = static_cast<uint8_t>( token_bytes_ );
        current_file_hash_ ^= CityHash64(data_end, kDataEndBytes - 8);
        write_uint64_le(data_end + kDataEndBytes - 8, current_file_hash_);
        current_file_.write(data_end, kDataEndBytes);
    }

    if (current_file_.fail() || current_index_.fail()) {
        LOG_ERROR() << "Failed to write tail on files";
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

void TokenizedDataPrep::Start(
    const std::string& data_folder_path,
    uint32_t token_bytes)
{
    data_folder_path_ = data_folder_path;
    token_bytes_ = token_bytes;

    current_file_number_ = 0;
    worker_error_ = false;
    stopped_ = false;

    pool_.Start();
    contexts_.resize(pool_.GetWorkerCount());
    for (int i = 0; i < (int)contexts_.size(); ++i) {
        contexts_[i] = std::make_shared<CompressorContext>(token_bytes_);
    }
}

bool TokenizedDataPrep::WriteTokens(
    const void* tokens,
    uint32_t token_count)
{
    total_tokens_ += token_count;

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

    std::shared_ptr<TokenizedBuffer> buffer = allocator_.Allocate(tokens, token_count * token_bytes_);
    pool_.QueueTask([this, on_file_start, buffer, token_count](int worker_index) {
        bool r = contexts_[worker_index]->WriteTokens(
            buffer->Data.data(),
            token_count,
            on_file_start);
        if (!r) {
            LOG_ERROR() << "Worker encountered an error";
            worker_error_ = true;
        }
        allocator_.Free(buffer);
    }, max_active_tasks, true/*round robin*/);

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
        LOG_ERROR() << "Failed to open index file: " << index_file_path;
        return false;
    }

    global_index_file << "# Generated by tokenized_data_prep.cpp from https://github.com/catid/lllm" << std::endl;
    global_index_file << "version: " << DATALOADER_VERSION << std::endl;
    global_index_file << "token_bytes: " << token_bytes_ << std::endl;
    global_index_file << "total_tokens: " << total_tokens_ << std::endl;

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
