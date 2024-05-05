#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>

#include "dataloader.hpp"
#include "worker_pool.hpp"
#include "compressor.hpp"

#include <yaml.hpp>

static const uint64_t kMaxFileSize = 4ULL * 1024 * 1024 * 1024;


//------------------------------------------------------------------------------
// TokenizedDataPrep

class TokenizedDataPrep {
public:
    TokenizedDataPrep(const std::string& data_folder_path);

    bool WriteTokenizedText(const uint16_t* tokenized_text, uint32_t text_length);
    bool Finalize();

private:
    std::string data_folder_path_ = ".";

    int current_file_number_ = 0;

    std::vector<uint8_t> packed_buffer_;

    Compressor compressor;

    uint64_t current_file_size_ = 0;

    std::ofstream global_index_;
    std::ofstream current_file_;
    std::ofstream current_index_;
    uint64_t current_file_hash_ = 0;
    uint64_t current_index_hash_ = 0;

    WorkerPool pool_;
};
