#include "tokenized_data_prep.hpp"

#include <city.h>
#include <cpppath.h>


//------------------------------------------------------------------------------
// CompressorWorker

void CompressorWorker::Start(
    OnFileComplete on_file_complete,
    OnFileStart on_file_start)
{
    on_file_complete_ = on_file_complete;
    on_file_start_ = on_file_start;
    worker_ = std::make_shared<ThreadWorker>();
}

bool CompressorWorker::Stop()
{
    worker_ = nullptr;
}

bool CompressorWorker::WriteTokenizedText(
    const uint32_t* tokenized_text,
    uint32_t text_length)
{

}


//------------------------------------------------------------------------------
// TokenizedDataPrep

TokenizedDataPrep::TokenizedDataPrep(const std::string& data_folder_path)
    : data_folder_path_(data_folder_path) {}

bool TokenizedDataPrep::WriteTokenizedText(
    const uint32_t* tokenized_text,
    uint32_t text_length)
{
    if (current_file_size_ >= kMaxFileSize) {
        if (!Finalize()) {
            return false;
        }
    }
#if 0
    // Split high and low bytes of the tokens
    packed_buffer_.resize(text_length * sizeof(uint16_t));
    uint8_t* low_buffer = packed_buffer_.data();
    uint8_t* high_buffer = packed_buffer_.data() + text_length;
    for (uint32_t i = 0; i < text_length; ++i) {
        low_buffer[i] = static_cast<uint8_t>( tokenized_text[i] );
        high_buffer[i] = static_cast<uint8_t>( tokenized_text[i] >> 8 );
    }

    if (!compressor.Compress(packed_buffer_.data(), text_length * sizeof(uint16_t))) {
        return false;
    }

    if (current_file_size_ == 0) {
        std::string index_file_name = "index_" + std::to_string(current_file_number_) + ".bin";
        std::string index_file_path = cpppath::join({data_folder_path_, index_file_name});

        current_index_.open(index_file_path, std::ios::binary);
        if (!current_index_) {
            return false;
        }

        std::string data_file_name = "data_" + std::to_string(current_file_number_) + ".bin";
        std::string data_file_path = cpppath::join({data_folder_path_, data_file_name});

        current_file_.open(data_file_path, std::ios::binary);
        if (!current_file_) {
            return false;
        }
    }

    // Write the offset of the current chunk to the index file
    uint32_t current_offset = static_cast<uint32_t>( current_file_size_ );
    current_index_.write(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));

    current_index_hash_ ^= CityHash64(reinterpret_cast<const char*>(&current_offset), sizeof(current_offset));

    // Write the compressed data to the current file
    current_file_.write(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());
    current_file_size_ += compressor.Result.size();

    current_file_hash_ ^= CityHash64(reinterpret_cast<const char*>(compressor.Result.data()), compressor.Result.size());

    if (current_file_.fail() || current_index_.fail()) {
        return false;
    }
#endif
    return true;
}

bool TokenizedDataPrep::Finalize() {
    if (current_file_size_ <= 0) {
        return true;
    }

#if 0
    // Last 64 bits is the file hash for verification
    current_file_.write(reinterpret_cast<const char*>(&current_file_hash_), sizeof(current_file_hash_));
    current_index_.write(reinterpret_cast<const char*>(&current_index_hash_), sizeof(current_index_hash_));

    bool success = true;
    if (current_file_.fail() || current_index_.fail()) {
        success = false;
    }

    current_file_.close();
    current_index_.close();
    current_file_hash_ = 0;
    current_index_hash_ = 0;

    current_file_number_++;
    current_file_size_ = 0;
    return success;
#endif
    return true;
}
