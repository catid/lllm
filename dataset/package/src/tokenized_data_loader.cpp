#include "tokenized_data_loader.hpp"

#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>

#include <mapped_file.hpp>

#include "tools.hpp"

//------------------------------------------------------------------------------
// RowReadResults

std::shared_ptr<Decompressor> RowReadResults::AddResult(
    uint32_t read_offset,
    uint32_t write_offset,
    uint32_t write_count)
{
    std::lock_guard<std::mutex> lock(Lock);

    std::shared_ptr<Decompressor> decompressor;

    if (Freed.empty()) {
        decompressor = std::make_shared<Decompressor>();
    } else {
        decompressor = Freed.back();
        Freed.pop_back();
        decompressor->Result.clear();
    }

    Results.push_back({decompressor, read_offset, write_offset, write_count});

    return decompressor;
}

void RowReadResults::Reset()
{
    std::lock_guard<std::mutex> lock(Lock);

    // Reuse all decompressor objects
    for (auto& result : Results) {
        Freed.push_back(result.Decompressor);
    }
    Results.clear();
    NextWriteOffset = 0;
}

uint32_t RowReadResults::WriteOutput(
    int32_t* output_row,
    uint32_t context_size,
    int32_t padding_token)
{
    std::lock_guard<std::mutex> lock(Lock);

    uint32_t max_token_count = 0;
    NextResults.clear();

    const uint32_t result_count = static_cast<uint32_t>( Results.size() );
    for (uint32_t i = 0; i < result_count; ++i) {
        const auto& result = Results[i];
        const auto& dest = result.Dest;
        const int32_t* decompressed_ptr = result.Decompressor->Result.data();

        memcpy(output_row + dest.WriteOffset, decompressed_ptr + dest.ReadOffset, dest.WriteCount * sizeof(int32_t));

        const uint32_t write_end = dest.WriteOffset + dest.WriteCount;
        max_token_count = std::max(max_token_count, write_end);

        if (write_end < context_size) {
            output_row[write_end] = padding_token;
        }

        // If there is more data left over:
        const uint32_t completed = dest.WriteCount + dest.ReadOffset;
        const uint32_t decompressed_tokens = static_cast<uint32_t>( result.Decompressor->Result.size() );
        if (completed < decompressed_tokens) {
            // FIXME: REVISIT THIS
            // Prepare to read from it again on the next iteration
            ReadResult next;
            next.Decompressor = result.Decompressor;
            next.Dest.ReadOffset = completed;
            next.Dest.WriteOffset = 0;
            next.Dest.WriteCount = std::min(context_size, decompressed_tokens - completed);
            NextResults.push_back(next);

            NextWriteOffset += next.Dest.WriteCount + 1;
        } else {
            Freed.push_back(result.Decompressor);
        }
    }

    for (uint32_t i = max_token_count; i < context_size; ++i) {
        output_row[i] = padding_token;
    }

    if (NextResults.size() > 1) {
        LOG_ERROR() << "Internal error: More than one ReadResult continuation!  This should never happen.";
    }

    // Results = NextResults
    std::swap(Results, NextResults);

    return max_token_count;
}


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

    const uint8_t* span_data = index_reader->GetData();
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

bool DataShardContext::GetSpan(
    uint32_t span_index,
    uint64_t& offset_out,
    uint32_t& cbytes_out,
    uint32_t& original_tokens_out)
{
    if (span_index >= NumSpans) {
        LOG_ERROR() << "Internal error: Requested span index " << span_index << " is out of range";
        return false;
    }

    uint32_t index = EpochSpanData[span_index];

    if (index * 4 >= IndexFile->GetSize()) {
        LOG_ERROR() << "Internal error: Requested index " << index << " is out of range";
        return false;
    }

    const uint8_t* span_data = IndexFile->GetData() + index * kIndexRecordBytes;
    const uint32_t start = read_uint32_le(span_data);
    const uint32_t original_tokens = read_uint32_le(span_data + 4);
    const uint32_t end = read_uint32_le(span_data + kIndexRecordBytes);

    if (start >= end) {
        LOG_ERROR() << "Internal error: Span index " << span_index << " has invalid start=" << start << " end=" << end;
        return false;
    }

    offset_out = start;
    cbytes_out = end - start;
    original_tokens_out = original_tokens;
    return true;
}


//------------------------------------------------------------------------------
// TokenizedDataLoader

bool TokenizedDataLoader::Start(const std::string& data_folder_path)
{
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

    epoch_config_ = EpochConfig(); // Defaults

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
    // Kill the prefill tasks
    pool_.Stop();

    // This kills the trailing read requests
    shards_.clear();

    row_read_results_.clear();
    global_index_yaml_ = nullptr;
}

void TokenizedDataLoader::StartEpoch(const EpochConfig& config)
{
    epoch_config_ = config;

    if (epoch_config_.LocalRank >= epoch_config_.LocalRankCount) {
        LOG_ERROR() << "Local rank " << config.LocalRank << " is out of range (local_ranks=" << config.LocalRankCount << ")";
        return;
    }

    LOG_INFO() << "Epoch shuffling " << epoch_config_.MicroBatchSize << " microbatches of " << epoch_config_.ContextSize << " tokens each"; 

    if (!WaitUntilDataReady()) {
        return;
    }

    // Clear decompressor state etc
    ResetPrefill();

    // Shuffle the order in which we step through the shards
    pool_.QueueTask([this](int /*worker_index*/) {
        shard_order_.resize(shards_.size() * kShardOrderMult);
        uint32_t* shard_ptr = shard_order_.data();

        // Repeat each shard number kShardOrderMult times
        for (uint32_t j = 0; j < (uint32_t)shards_.size(); ++j) {
            uint32_t* j_ptr = shard_ptr + j * kShardOrderMult;
            for (uint32_t i = 0; i < kShardOrderMult; ++i) {
                j_ptr[i] = j;
            }
        }

        std::seed_seq seed{epoch_config_.Seed0, epoch_config_.Seed1};
        std::mt19937 rng(seed);
        std::shuffle(shard_order_.begin(), shard_order_.end(), rng);
    });

    uint64_t seed1 = epoch_config_.Seed1;

    // Shuffle all the shard indices
    for (auto& shard : shards_) {
        pool_.QueueTask([this, shard, seed1](int /*worker_index*/) {
            shard->ShuffleIndices(epoch_config_.Seed0, seed1, epoch_config_.LocalRank, epoch_config_.LocalRankCount);
        });
        seed1++;
    }

    // Wait for shuffles to complete before prefilling
    pool_.WaitForTasks();

    // Calculate the total number of steps in the dataset
    CalculateTotalSteps();

    if (epoch_config_.StartStep > 0) {
        LOG_INFO() << "Resuming training from step " << epoch_config_.StartStep;

        for (int i = 0; i < epoch_config_.StartStep; ++i) {
            GenerateMicrobatchRequests(true);
        }

        current_step_ = epoch_config_.StartStep;
    }

    Prefill();
}

void TokenizedDataLoader::ResetPrefill()
{
    next_shard_index_ = 0;
    prefill_inflight_ = 0;
    prefill_complete_ = false;
    prefill_started_ = false;
    current_step_ = 0;

    // Resize to the number of prefill tasks
    row_read_results_.resize(epoch_config_.MicroBatchSize);
    for (uint32_t batch_index = 0; batch_index < epoch_config_.MicroBatchSize; ++batch_index) {
        row_read_results_[batch_index].Reset();
    }
}

void TokenizedDataLoader::PostRequests()
{
    if (microbatch_requests_.empty()) {
        prefill_complete_ = true;
        return;
    }

    prefill_complete_ = false;
    prefill_started_ = true;
    prefill_inflight_ += (uint32_t)microbatch_requests_.size();

    for (auto request : microbatch_requests_) {
        pool_.QueueTask([this, request](int /*worker_index*/) {
            if (request.shard_index >= shards_.size()) {
                LOG_ERROR() << "Internal error: Requested shard index " << request.shard_index << " is out of range";
                worker_error_ = true;
                return;
            }
            if (request.batch_index >= row_read_results_.size()) {
                LOG_ERROR() << "Internal error: Requested batch index " << request.batch_index << " is out of range";
                worker_error_ = true;
                return;
            }

            // Read offsets from mmap index file
            uint64_t offset = 0;
            uint32_t cbytes = 0;
            uint32_t original_tokens = 0;
            bool span_okay = shards_[request.shard_index]->GetSpan(
                request.shard_span_index,
                offset,
                cbytes,
                original_tokens);

            if (!span_okay) {
                LOG_ERROR() << "Invalid span data for shard=" << request.shard_index << " index=" << request.shard_span_index;
                worker_error_ = true;
                return;
            }

            shards_[request.shard_index]->DataFile->Read(offset, cbytes,
                [this, cbytes, original_tokens, offset, request](
                    uint8_t* compressed_data,
                    uint32_t compressed_bytes)
            {
                if (request.batch_index >= row_read_results_.size()) {
                    LOG_ERROR() << "Internal error: Requested batch index " << request.batch_index << " is out of range";
                    worker_error_ = true;
                    return;
                }

                total_disk_read_ += compressed_bytes;

                auto& decompressor = decompressors_[request.batch_index];

                bool r = decompressor->Decompress(
                    compressed_data,
                    compressed_bytes,
                    original_tokens,
                    token_bytes_);
                if (!r) {
                    LOG_ERROR() << "Failed to decompress data for shard=" << request.shard_index << " index=" << request.shard_span_index;
                    worker_error_ = true;
                }

                total_decompressed_tokens_ += decompressor->Result.size();

                // This is used only for skipping
                output_used_[request.batch_index] = request.skip;

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

int TokenizedDataLoader::GenerateMicrobatchRequests(bool skipping)
{
    int num_requests = 0;

    // FIXME: We need to keep the previous request if we didn't actually make the request
    microbatch_requests_.clear();

    for (uint32_t batch_index = 0; batch_index < epoch_config_.MicroBatchSize; ++batch_index)
    {
        auto& row = row_read_results_[batch_index];

        uint32_t next_write_offset = row.RemainingCount + 1;

        // While there is space for data:
        while (next_write_offset < epoch_config_.ContextSize)
        {
            const uint32_t remaining_space = epoch_config_.ContextSize - next_write_offset;
            if (remaining_space < epoch_config_.MinDataLength) {
                break; // No more space for meaningful data
            }

            ReadRequest request;
            request.batch_index = batch_index;
            if (!NextSpan(request)) {
                break; // No more data to read
            }
            request.dest.ReadOffset = 0;
            request.dest.WriteOffset = next_write_offset;

            // Get original token count
            uint64_t offset = 0;
            uint32_t cbytes = 0, original_tokens = 0;
            bool r = shards_[request.shard_index]->GetSpan(
                request.shard_span_index,
                offset,
                cbytes,
                original_tokens);
            if (!r) {
                LOG_ERROR() << "Failed to read span " << request.shard_span_index
                    << " from shard " << request.shard_index;
                return;
            }

            uint32_t write_end = request.dest.WriteOffset + original_tokens;
            if (write_end > epoch_config_.ContextSize) {
                request.dest.WriteCount = epoch_config_.ContextSize - request.dest.WriteOffset;
                next_write_offset = epoch_config_.ContextSize; // Will break after this one

                // Store remaining count for this row for next step
                row.RemainingCount = original_tokens - request.dest.WriteCount;
            } else {
                request.dest.WriteCount = original_tokens;
                next_write_offset += write_end + 1;
            }

            if (!skipping) {
                microbatch_requests_.push_back(request);
            }
            ++num_requests;
        }
    }

    return num_requests;
}

void TokenizedDataLoader::CalculateTotalSteps()
{
    total_steps_ = 0;

    for (;;) {
        int num_requests = GenerateMicrobatchRequests(true);
        if (num_requests <= 0) {
            break;
        }

        ++total_steps_;
    }

    // Reset state messed with by NextSpan()
    next_shard_index_ = 0;
    for (const auto& shard : shards_) {
        shard->EpochNextSpan = 0;
    }
}

void TokenizedDataLoader::Prefill() {
    LOG_DEBUG() << "Prefilling " << epoch_config_.MicroBatchSize << "...";

    if (prefill_inflight_ > 0) {
        LOG_ERROR() << "Internal error: Prefill still inflight"; 
        return;
    }

    GenerateMicrobatchRequests(false);
    PostRequests();
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
    uint8_t* is_continuation,
    uint32_t* step_out,
    uint32_t* total_steps_out)
{
    if (!WaitUntilDataReady()) {
        LOG_DEBUG() << "Data did not become ready";
        return false;
    }

    if (step_out) {
        *step_out = current_step_++;
    }
    if (total_steps_out) {
        *total_steps_out = total_steps_;
    }

    if (worker_error_) {
        LOG_ERROR() << "Data loader encountered an error.  Possibly data file is corrupted.";
        return false;
    }

    uint32_t max_token_count = 0;
    uint32_t num_rows = 0;

    for (uint32_t batch_index = 0; batch_index < epoch_config_.MicroBatchSize; ++batch_index, output_batch += epoch_config_.ContextSize)
    {
        int32_t* output_row = output_batch;
        auto& row = row_read_results_[batch_index];

        *is_continuation = row.IsContinuation();

        const uint32_t written = row.WriteOutput(
            output_row,
            epoch_config_.ContextSize,
            epoch_config_.PaddingToken);
        if (written == 0) {
            continue; // No data to write
        }

        max_token_count = std::max(max_token_count, written);
        output_row += epoch_config_.ContextSize;
        ++num_rows;
        ++is_continuation;
    }

    // Pad the remaining unwritten rows with padding tokens
    if (num_rows < epoch_config_.MicroBatchSize) {
        const uint32_t pad_rows = epoch_config_.MicroBatchSize - num_rows;
        for (uint32_t i = 0; i < epoch_config_.ContextSize; ++i) {
            output_batch[i] = epoch_config_.PaddingToken;
        }
        memset(is_continuation, 0, pad_rows);
    }

    if (micro_batch_out) {
        *micro_batch_out = num_rows;
    }
    if (max_tokens_out) {
        *max_tokens_out = max_token_count;
    }

    if (num_rows == 0) {
        LOG_INFO() << "GetTokenArray: Training data exhausted.  Disk compression: " << (total_disk_read_ / total_decompressed_tokens_) << " bytes/token";
        return false;
    }

    Prefill();
    return true;
}
