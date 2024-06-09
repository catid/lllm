#include "tokenized_data_loader.hpp"

#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>

#include <mapped_file.hpp>

#include "tools.hpp"


//------------------------------------------------------------------------------
// DataShardContext

bool DataShardContext::Open(
    const std::string& data_file_path,
    const std::string& index_file_path,
    uint32_t token_bytes)
{
    token_bytes_ = token_bytes;

    std::shared_ptr<AsyncUringReader> data_reader = std::make_shared<AsyncUringReader>();
    if (!data_reader->Open(data_file_path)) {
        LOG_INFO() << "Failed to open data file at " << data_file_path;
        return false;
    }

    std::shared_ptr<MappedFileReader> index_reader = std::make_shared<MappedFileReader>();
    if (!index_reader->Open(index_file_path)) {
        LOG_INFO() << "Failed to open index file at " << index_file_path;
        return false;
    }

    uint64_t index_file_bytes = index_reader->GetSize();
    if (index_file_bytes < 8) {
        LOG_INFO() << "Index file is too small at " << index_file_path;
        return false;
    }

    uint64_t num_spans = (index_file_bytes - kIndexEndBytes) / kIndexRecordBytes;
    if (num_spans > UINT32_MAX) {
        LOG_INFO() << "Too many regions in index file at " << index_file_path;
        return false;
    }

    const uint8_t* span_data = IndexFile->GetData();
    const uint32_t data_file_bytes = read_uint32_le(span_data + index_file_bytes - kIndexEndBytes);
    const uint32_t index_version = span_data[index_file_bytes - kIndexEndBytes + 4];
    const uint32_t index_token_bytes = span_data[index_file_bytes - kIndexEndBytes + 5];

    if (index_version != DATALOADER_VERSION) {
        LOG_ERROR() << "Index file version mismatch: expected " << DATALOADER_VERSION << ", found " << index_version;
        return false;
    }
    if (token_bytes != index_token_bytes) {
        LOG_ERROR() << "Index file token_bytes mismatch: expected " << token_bytes << ", found " << index_token_bytes;
        return false;
    }
    if (data_file_bytes + kDataEndBytes != data_reader->GetSize()) {
        LOG_ERROR() << "Data file size mismatch: expected " << data_file_bytes + kDataEndBytes << ", found " << data_reader->GetSize();
        return false;
    }

    NumSpans = num_spans;
    DataFile = data_reader;
    IndexFile = index_reader;
    return true;
}

void DataShardContext::Close()
{
    DataFile = nullptr;
    IndexFile = nullptr;
}

void DataShardContext::ShuffleIndices(
    uint64_t seed0, uint64_t seed1,
    uint32_t rank,
    uint32_t local_ranks)
{
    ShuffledIndices.resize(NumSpans);
    uint32_t* span_ptr = ShuffledIndices.data();
    for (uint32_t i = 0; i < NumSpans; ++i) {
        span_ptr[i] = i;
    }

    std::seed_seq seed{seed0, seed1};
    std::mt19937 rng(seed);
    std::shuffle(ShuffledIndices.begin(), ShuffledIndices.end(), rng);

    // Apply local rank subset:

    uint32_t max_rank_count = (NumSpans + local_ranks - 1) / local_ranks;
    uint32_t local_rank_start = max_rank_count * rank;

    EpochSpanData = ShuffledIndices.data() + local_rank_start;

    // Calculate number of spans to read for this epoch
    if (local_rank_start + max_rank_count < NumSpans) {
        EpochSpanCount = max_rank_count;
    } else {
        EpochSpanCount = NumSpans - local_rank_start;
    }
    EpochNextSpan = 0;
}

uint64_t DataShardContext::GetSpan(
    uint32_t span_index,
    uint32_t& cbytes_out,
    uint32_t& original_bytes_out)
{
    if (span_index >= NumSpans) {
        LOG_ERROR() << "Internal error: Requested span index " << span_index << " is out of range";
        return 0;
    }

    uint32_t index = EpochSpanData[span_index];

    if (index * 4 >= IndexFile->GetSize()) {
        LOG_ERROR() << "Internal error: Requested index " << index << " is out of range";
        return 0;
    }

    const uint8_t* span_data = IndexFile->GetData() + index * kIndexRecordBytes;
    const uint32_t start = read_uint32_le(span_data);
    const uint32_t original_bytes = read_uint32_le(span_data + 4);
    const uint32_t end = read_uint32_le(span_data + kIndexRecordBytes);

    if (start >= end) {
        LOG_ERROR() << "Internal error: Span index " << span_index << " has invalid start=" << start << " end=" << end;
        return 0;
    }

    cbytes_out = end - start;
    original_bytes_out = original_bytes;
    return start;
}


//------------------------------------------------------------------------------
// TokenizedDataLoader

bool TokenizedDataLoader::Start(
    const std::string& data_folder_path,
    uint32_t rank,
    uint32_t local_ranks)
{
    rank_ = rank;
    local_ranks_ = local_ranks;

    if (rank_ >= local_ranks_) {
        LOG_ERROR() << "Rank " << rank_ << " is out of range (local_ranks=" << local_ranks_ << ")";
        return false;
    }
    if (local_ranks <= 0) {
        LOG_ERROR() << "Local ranks must be greater than 0";
        return false;
    }

    Stop();

    global_index_yaml_ = std::make_shared<GlobalIndexYaml>();
    if (!global_index_yaml_->Read(data_folder_path)) {
        LOG_INFO() << "Failed to read global index file at " << data_folder_path;
        return false;
    }

    if (global_index_yaml_->version_ != DATALOADER_VERSION) {
        LOG_ERROR() << "Global index file version mismatch: expected " << DATALOADER_VERSION << ", found " << global_index_yaml_->version_;
        return false;
    }

    token_bytes_ = static_cast<uint32_t>( global_index_yaml_->token_bytes_ );

    pool_.Start();

    // These are set during StartEpoch()
    context_size_ = 0;
    micro_batch_size_ = 0;

    ResetPrefill();

    // Load data index files in parallel:

    worker_error_ = false;
    shards_ready_ = 0;

    const int num_files = (int)global_index_yaml_->data_files_.size();
    for (int i = 0; i < num_files; ++i) {
        const std::string& data_file_path = global_index_yaml_->data_files_[i];
        const std::string& index_file_path = global_index_yaml_->index_files_[i];

        auto shard = std::make_shared<DataShardContext>();
        shards_.push_back(shard);

        pool_.QueueTask([this, shard, data_file_path, index_file_path](int /*worker_index*/) {
            if (worker_error_) {
                return;
            }
            if (!shard->Open(data_file_path, index_file_path, token_bytes_)) {
                LOG_INFO() << "Failed to open data file at " << data_file_path;
                worker_error_ = true;
            }
        });
    };

    pool_.WaitForTasks();

    if (worker_error_) {
        LOG_INFO() << "Failed to open all data files";
        return false;
    }

    return true;
}

void TokenizedDataLoader::Stop()
{
    pool_.Stop();

    decompressors_.clear();
    output_used_.clear();
    shards_.clear();
    global_index_yaml_ = nullptr;
}

void TokenizedDataLoader::StartEpoch(
    uint64_t seed0,
    uint64_t seed1,
    uint32_t micro_batch_size,
    uint32_t context_size)
{
    LOG_INFO() << "Epoch shuffling " << micro_batch_size << " microbatches of " << context_size << " tokens each";

    if (!WaitUntilDataReady()) {
        return;
    }

    micro_batch_size_ = micro_batch_size;
    context_size_ = context_size;

    // Clear decompressor state etc
    ResetPrefill();

    // Shuffle the order in which we step through the shards
    pool_.QueueTask([this, seed0, seed1](int /*worker_index*/) {
        shard_order_.resize(shards_.size() * kShardOrderMult);
        uint32_t* shard_ptr = shard_order_.data();

        // Repeat each shard number kShardOrderMult times
        for (uint32_t j = 0; j < (uint32_t)shards_.size(); ++j) {
            uint32_t* j_ptr = shard_ptr + j * kShardOrderMult;
            for (uint32_t i = 0; i < kShardOrderMult; ++i) {
                j_ptr[i] = j;
            }
        }

        std::seed_seq seed{seed0, seed1};
        std::mt19937 rng(seed);
        std::shuffle(shard_order_.begin(), shard_order_.end(), rng);
    });
    seed1++;

    // Shuffle all the shard indices
    for (auto& shard : shards_) {
        pool_.QueueTask([this, shard, seed0, seed1](int /*worker_index*/) {
            shard->ShuffleIndices(seed0, seed1, rank_, local_ranks_);
        });
        seed1++;
    }

    // Wait for shuffles to complete before prefilling
    pool_.WaitForTasks();

    Prefill();
}

void TokenizedDataLoader::ResetPrefill()
{
    next_shard_index_ = 0;
    prefill_inflight_ = 0;
    prefill_complete_ = false;
    prefill_started_ = false;

    // Resize to the number of prefill tasks
    decompressors_.resize(micro_batch_size_);
    output_used_.resize(micro_batch_size_);
    for (uint32_t batch_index = 0; batch_index < micro_batch_size_; ++batch_index) {
        if (!decompressors_[batch_index]) {
            decompressors_[batch_index] = std::make_shared<Decompressor>();
        }
        output_used_[batch_index] = 0;
        decompressors_[batch_index]->Result.clear();
    }
}

void TokenizedDataLoader::Prefill() {
    LOG_DEBUG() << "Prefilling " << micro_batch_size_ << "...";

    if (prefill_inflight_ > 0) {
        LOG_ERROR() << "Internal error: Prefill still inflight"; 
        return;
    }

    std::vector<ReadRequest> requests;

    int continuations = 0;

    for (uint32_t batch_index = 0; batch_index < micro_batch_size_; ++batch_index)
    {
        auto& decompressor = decompressors_[batch_index];
        const uint32_t decompressed_words = decompressor->Result.size() / 4;
        const uint32_t used = output_used_[batch_index];

        // If there is still data to read, do not prefill this row
        if (used > 0 && used < decompressed_words) {
            ++continuations;
            continue;
        }

        ReadRequest request;
        request.batch_index = batch_index;
        if (!NextSpan(request)) {
            break; // No more data to read
        }

        requests.push_back(request);

        // Reset the used count for this row
        output_used_[batch_index] = 0;
    }

    if (requests.empty()) {
        if (continuations == 0) {
            LOG_INFO() << "Prepared dataset files contain no more data";
        }
        prefill_complete_ = true;
        return;
    }

    prefill_complete_ = false;
    prefill_started_ = true;
    prefill_inflight_ += (uint32_t)requests.size();

    for (auto request : requests) {
        pool_.QueueTask([this, request](int /*worker_index*/) {
            if (request.shard_index >= shards_.size()) {
                LOG_ERROR() << "Internal error: Requested shard index " << request.shard_index << " is out of range";
                worker_error_ = true;
                return;
            }
            if (request.batch_index >= decompressors_.size()) {
                LOG_ERROR() << "Internal error: Requested batch index " << request.batch_index << " is out of range";
                worker_error_ = true;
                return;
            }

            // Read offsets from mmap index file
            uint32_t cbytes = 0;
            uint32_t original_bytes = 0;
            uint64_t offset = shards_[request.shard_index]->GetSpan(
                request.shard_span_index,
                cbytes,
                original_bytes);

            if (cbytes == 0) {
                LOG_ERROR() << "Invalid span data for shard=" << request.shard_index << " index=" << request.shard_span_index;
                worker_error_ = true;
                return;
            }

            shards_[request.shard_index]->DataFile->Read(offset, cbytes,
                [this, cbytes, original_bytes, offset, request](
                    uint8_t* compressed_data,
                    uint32_t compressed_bytes)
            {
                if (request.batch_index >= decompressors_.size()) {
                    LOG_ERROR() << "Internal error: Requested batch index " << request.batch_index << " is out of range";
                    worker_error_ = true;
                    return;
                }

                total_disk_read_ += compressed_bytes;

                auto& decompressor = decompressors_[request.batch_index];
                bool r = decompressor->Decompress(
                    compressed_data,
                    compressed_bytes,
                    original_bytes,
                    token_bytes_);
                if (!r) {
                    LOG_ERROR() << "Failed to decompress data for shard=" << request.shard_index << " index=" << request.shard_span_index;
                    worker_error_ = true;
                }

                total_decompressed_bytes_ += decompressor->Result.size();

                //LOG_INFO() << "Decompressed " << decompressor->Result.size() << " bytes from " << cbytes << " for shard=" << request.shard_index << " index=" << request.shard_span_index << " offset=" << offset;

                uint32_t remaining = --prefill_inflight_;
                if (remaining == 0) {
                    // All prefill requests have completed
                    std::unique_lock<std::mutex> lock(output_mutex_);
                    prefill_complete_ = true;
                    output_condition_.notify_all();
                }
            });

        });
    }
}

bool TokenizedDataLoader::NextSpan(ReadRequest& request) {
    // TBD: This will do more than one round-robin through the shards due to the randomization
    // in the shard order.
    for (uint32_t tries = 0; tries < (uint32_t)shard_order_.size(); ++tries)
    {
        const uint32_t shard_index = shard_order_[next_shard_index_++];

        // Round-robin through the shards
        if (next_shard_index_ >= (uint32_t)shard_order_.size()) {
            next_shard_index_ = 0;
        }

        auto& shard = shards_[shard_index];
        uint32_t shard_span_index = shard->EpochNextSpan;

        if (shard_span_index < shard->EpochSpanCount) {
            shard->EpochNextSpan = shard_span_index + 1;

            request.shard_index = shard_index;
            request.shard_span_index = shard_span_index;
            return true;
        }
    }

    // Data exhausted
    return false;
}

bool TokenizedDataLoader::WaitUntilDataReady() {
    std::unique_lock<std::mutex> lock(output_mutex_);
    output_condition_.wait(lock, [this]{ return Terminated || !prefill_started_ || prefill_complete_; });
    return !Terminated;
}

bool TokenizedDataLoader::GetTokenArray(
    uint32_t* micro_batch_out,
    uint32_t* max_tokens_out,
    int32_t* output_batch,
    uint8_t* is_continuation)
{
    if (!WaitUntilDataReady()) {
        LOG_DEBUG() << "Data did not become ready";
        return false;
    }

    if (worker_error_) {
        LOG_ERROR() << "Data loader encountered an error.  Possibly data file is corrupted.";
        return false;
    }

    uint32_t max_token_count = 0;
    uint32_t num_rows = 0;

    for (uint32_t batch_index = 0; batch_index < micro_batch_size_; ++batch_index) {
        auto& decompressor = decompressors_[batch_index];
        if (!decompressor) {
            continue;
        }
        const uint32_t* decompressed_ptr = reinterpret_cast<const uint32_t*>( decompressor->Result.data() );
        const uint32_t decompressed_words = decompressor->Result.size() / 4;

        const uint32_t used = output_used_[batch_index];
        const uint32_t available = decompressed_words - used;
        if (available <= 0) {
            continue;
        }
        uint32_t copy_words = std::min(available, context_size_);

        LOG_DEBUG() << "batch_index=" << batch_index << ", copy_words=" << copy_words << ", used=" << used << ", available=" << available << ", decompressed_words=" << decompressed_words;

        memcpy(output_batch, decompressed_ptr + used, copy_words * sizeof(uint32_t));
        output_batch += copy_words;
        max_token_count = std::max(max_token_count, copy_words);

        const uint32_t pad_words = context_size_ - copy_words;
        if (pad_words > 0) {
            for (uint32_t i = 0; i < pad_words; ++i) {
                output_batch[i] = kPaddingToken;
            }
            output_batch += pad_words;
        }

        // Signal continuation for this row
        *is_continuation++ = (used != 0);

        output_used_[batch_index] = used + copy_words;

        ++num_rows;
    }

    // Pad the remaining unwritten rows with padding tokens
    if (num_rows < micro_batch_size_) {
        const uint32_t pad_rows = micro_batch_size_ - num_rows;
        for (uint32_t i = 0; i < context_size_; ++i) {
            output_batch[i] = kPaddingToken;
        }
        memset(is_continuation, 0, pad_rows);
    }

    *micro_batch_out = num_rows;
    *max_tokens_out = max_token_count;

    if (num_rows == 0) {
        LOG_INFO() << "GetTokenArray: Training data exhausted.  Disk compression: " << (total_disk_read_ * 100.0 / total_decompressed_bytes_) << "% of original tokens"; 
        return false;
    }

    Prefill();
    return true;
}
