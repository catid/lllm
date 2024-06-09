/*
    Preparation for training data.
*/

#pragma once

#include "dataloader.hpp"
#include "worker_pool.hpp"
#include "compressor.hpp"
#include "tools.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <functional>

//------------------------------------------------------------------------------
// Constants

// Limit size to 2GB to avoid overflowing signed 32-bit integers
static const uint64_t kMaxFileSize = 0x7fffffff - 1000000;


//------------------------------------------------------------------------------
// CompressorContext

using OnFileStart = std::function<void(
    std::string& data_file_path,
    std::string& index_file_path)>;

class CompressorContext {
public:
    CompressorContext(uint32_t token_bytes) {
        token_bytes_ = token_bytes;
    }

    bool WriteTokens(
        const void* tokens,
        uint32_t token_count,
        OnFileStart on_file_start);

    bool FinishCurrentFile();

protected:
    uint32_t token_bytes_ = 0;

    std::string data_file_path_;
    std::ofstream current_file_;
    uint64_t current_file_hash_ = 0;

    std::string index_file_path_;
    std::ofstream current_index_;
    uint64_t current_index_hash_ = 0;
    uint64_t current_file_bytes_ = 0;

    Compressor compressor_;
};


//------------------------------------------------------------------------------
// TokenizedDataPrep

class TokenizedDataPrep {
public:
    ~TokenizedDataPrep() {
        Stop();
    }

    void Start(
        const std::string& data_folder_path,
        uint32_t token_bytes);
    bool WriteTokens(
        const void* tokens,
        uint32_t token_count);
    bool Stop();

private:
    std::string data_folder_path_ = ".";
    uint32_t token_bytes_ = 0;
    std::atomic<uint64_t> total_tokens_ = ATOMIC_VAR_INIT(0);

    std::vector<std::shared_ptr<CompressorContext>> contexts_;
    WorkerPool pool_;
    TokenizedAllocator allocator_;

    int current_file_number_ = 0;
    std::atomic<bool> worker_error_ = ATOMIC_VAR_INIT(false);

    std::mutex global_index_mutex_;
    std::vector<std::string> index_files_;
    std::vector<std::string> data_files_;

    bool stopped_ = false;
};
