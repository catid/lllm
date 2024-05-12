#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <functional>

#include "dataloader.hpp"
#include "worker_pool.hpp"
#include "compressor.hpp"
#include "tools.hpp"

//------------------------------------------------------------------------------
// Constants

// Limit size so that it is unlikely we will go over a 32-bit address space
static const uint64_t kMaxFileSize = 0xffffffff - 1000000;


//------------------------------------------------------------------------------
// CompressorContext

using OnFileStart = std::function<void(std::string& data_file_path, std::string& index_file_path)>;

class CompressorContext {
public:
    bool WriteTokenizedText(
        const uint32_t* tokenized_text,
        uint32_t text_length,
        OnFileStart on_file_start);

    bool FinishCurrentFile();

protected:
    std::string data_file_path_;
    std::ofstream current_file_;
    uint64_t current_file_hash_ = 0;

    std::string index_file_path_;
    std::ofstream current_index_;
    uint64_t current_index_hash_ = 0;
    uint64_t current_file_bytes_ = 0;

    uint32_t max_region_bytes_ = 0;

    Compressor compressor;
};


//------------------------------------------------------------------------------
// TokenizedDataPrep

class TokenizedDataPrep {
public:
    void Start(const std::string& data_folder_path);
    bool WriteTokenizedText(const uint32_t* tokenized_text, uint32_t text_length);
    bool Stop();

private:
    std::string data_folder_path_ = ".";

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
