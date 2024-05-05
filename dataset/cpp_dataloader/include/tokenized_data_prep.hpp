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

#include <yaml.hpp>

//------------------------------------------------------------------------------
// Constants

static const uint64_t kMaxFileSize = 4ULL * 1024 * 1024 * 1024;


//------------------------------------------------------------------------------
// CompressorWorker

using OnFileComplete = std::function<void()>;
using OnFileStart = std::function<void(std::string& data_file_path, std::string& index_file_path)>;

class CompressorWorker {
public:
    ~CompressorWorker() {
        Stop();
    }

    void Start(OnFileComplete on_file_complete, OnFileStart on_file_start);
    bool Stop();

    bool WriteTokenizedText(
        const uint32_t* tokenized_text,
        uint32_t text_length);

    int GetActiveTaskCount() const {
        if (!worker_) {
            return 0;
        }
        return worker_->GetActiveTaskCount();
    }

protected:
    OnFileComplete on_file_complete_;
    OnFileStart on_file_start_;

    std::shared_ptr<ThreadWorker> worker_;
    TokenizedAllocator allocator_;

    std::ofstream current_file_;
    std::ofstream current_index_;
    uint64_t current_file_hash_ = 0;
    uint64_t current_index_hash_ = 0;

    std::vector<uint8_t> packed_buffer_;
    Compressor compressor;
};


//------------------------------------------------------------------------------
// TokenizedDataPrep

class TokenizedDataPrep {
public:
    TokenizedDataPrep(const std::string& data_folder_path);

    bool WriteTokenizedText(const uint32_t* tokenized_text, uint32_t text_length);
    bool Finalize();

private:
    std::string data_folder_path_ = ".";

    int current_file_number_ = 0;

    uint64_t current_file_size_ = 0;

    std::ofstream global_index_;
};
