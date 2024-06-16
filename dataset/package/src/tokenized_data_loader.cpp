#include "tokenized_data_loader.hpp"

#include <algorithm>
#include <random>
#include <iostream>
#include <cstring>

#include <mapped_file.hpp>

#include "tools.hpp"

//------------------------------------------------------------------------------
// RowReadResults

std::shared_ptr<Decompressor> RowReadResults::AddResult(const BufferTarget& dest)
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

    Results.push_back({decompressor, dest});

    return decompressor;
}

void RowReadResults::Reset()
{
    std::lock_guard<std::mutex> lock(Lock);

    // Reuse all decompressor objects
    for (auto& result : Results) {
        Freed.push_back(result.Decomp);
    }
    Results.clear();
    RemainingCount = 0;
}

uint32_t RowReadResults::WriteOutput(int32_t* output_row, const EpochConfig& config)
{
    std::lock_guard<std::mutex> lock(Lock);

    uint32_t max_token_count = 0;

    // By default we have no carry-over data
    NextResults.clear();
    RemainingCount = 0;

    const uint32_t result_count = static_cast<uint32_t>( Results.size() );
    for (uint32_t i = 0; i < result_count; ++i) {
        const auto& result = Results[i];
        const auto& dest = result.Dest;
        const int32_t* decompressed_ptr = result.Decomp->Result.data();

        memcpy(output_row + dest.WriteOffset, decompressed_ptr + dest.ReadOffset, dest.WriteCount * sizeof(int32_t));

        const uint32_t write_end = dest.WriteOffset + dest.WriteCount;
        max_token_count = std::max(max_token_count, write_end);

        if (write_end < config.ContextSize) {
            output_row[write_end] = config.PaddingToken;
        }

        // If there is more data left over:
        const uint32_t completed = dest.WriteCount + dest.ReadOffset;
        const uint32_t decompressed_tokens = static_cast<uint32_t>( result.Decomp->Result.size() );
        const uint32_t remaining = decompressed_tokens - completed;

        // If there is some data left over:
        if (completed < decompressed_tokens && remaining >= config.MinDataLength) {
            // Prepare to read from it again on the next iteration
            ReadResult next;
            next.Decomp = result.Decomp;
            next.Dest.ReadOffset = completed;
            next.Dest.WriteOffset = 0;
            next.Dest.WriteCount = std::min(config.ContextSize, remaining);

            NextResults.push_back(next);
            RemainingCount = remaining; // Can be longer than ContextSize
        } else {
            Freed.push_back(result.Decomp);
        }
    }

    for (uint32_t i = max_token_count; i < config.ContextSize; ++i) {
        output_row[i] = config.PaddingToken;
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

    config_ = EpochConfig(); // Defaults

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

void TokenizedDataLoader::BeginEpoch(const EpochConfig& config)
{
    config_ = config;

    if (config_.LocalRank >= config_.LocalRankCount) {
        LOG_ERROR() << "Local rank " << config.LocalRank << " is out of range (local_ranks=" << config.LocalRankCount << ")";
        return;
    }

    LOG_DEBUG() << "Epoch shuffling " << config_.MicroBatchSize << " microbatches of " << config_.ContextSize << " tokens each"; 

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

        std::seed_seq seed{config_.Seed0, config_.Seed1};
        std::mt19937 rng(seed);
        std::shuffle(shard_order_.begin(), shard_order_.end(), rng);
    });

    uint64_t seed1 = config_.Seed1;

    // Shuffle all the shard indices
    for (auto& shard : shards_) {
        pool_.QueueTask([this, shard, seed1](int /*worker_index*/) {
            shard->ShuffleIndices(config_.Seed0, seed1, config_.LocalRank, config_.LocalRankCount);
        });
        seed1++;
    }

    // Wait for shuffles to complete before prefilling
    pool_.WaitForTasks();

    // Calculate the total number of steps in the dataset
    CalculateTotalSteps();

    if (config_.StartStep > 0) {
        LOG_INFO() << "Resuming training from step " << config_.StartStep;

        for (uint32_t i = 0; i < config_.StartStep; ++i) {
            GenerateMicrobatchRequests(true);
        }

        current_step_ = config_.StartStep;
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
    row_read_results_.resize(config_.MicroBatchSize);
    for (uint32_t batch_index = 0; batch_index < config_.MicroBatchSize; ++batch_index) {
        if (!row_read_results_[batch_index]) {
            row_read_results_[batch_index] = std::make_shared<RowReadResults>();
        }
        row_read_results_[batch_index]->Reset();
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

                auto decompressor = row_read_results_[request.batch_index]->AddResult(request.dest);

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

bool TokenizedDataLoader::GenerateMicrobatchRequests(bool skipping)
{
    int num_requests = 0;

    microbatch_requests_.clear();

    for (uint32_t batch_index = 0; batch_index < config_.MicroBatchSize; ++batch_index)
    {
        auto& row = row_read_results_[batch_index];

        // If there is some data to back up and read from when we were skipping:
        if (!skipping && row->SkipRequest.dest.WriteCount > 0) {
            microbatch_requests_.push_back(row->SkipRequest);
        }

        // Reset skip request
        row->SkipRequest.dest.WriteCount = 0;

        // If the data that is already waiting will fill the context:
        if (row->RemainingCount >= config_.ContextSize) {
            row->ExpectedNextRemainingCount = row->RemainingCount - config_.ContextSize;
            if (row->ExpectedNextRemainingCount < config_.MinDataLength) {
                row->ExpectedNextRemainingCount = 0;
            } else if (skipping) {
                row->SkipRequest.dest.WriteOffset = 0;
                row->SkipRequest.dest.ReadOffset += config_.ContextSize;
                row->SkipRequest.dest.WriteCount = std::min(config_.ContextSize, row->ExpectedNextRemainingCount);
            }
            continue;
        }

        row->ExpectedNextRemainingCount = 0; // Default is to have 0 remaining

        uint32_t next_write_offset = row->RemainingCount;
        if (next_write_offset > 0) {
            next_write_offset++;
        }
        //LOG_INFO() << "FIXME: batch_index=" << batch_index << " next_write_offset=" << next_write_offset;

        // While there is space for data:
        while (next_write_offset < config_.ContextSize)
        {
            const uint32_t remaining_space = config_.ContextSize - next_write_offset;
            if (remaining_space <=/*padding*/ config_.MinDataLength) {
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
                return false;
            }

            uint32_t write_end = request.dest.WriteOffset + original_tokens;
            if (write_end >= config_.ContextSize) {
                if (request.dest.WriteOffset >= config_.ContextSize) {
                    LOG_ERROR() << "Internal error: Requested write offset " << request.dest.WriteOffset << " is greater than context size " << config_.ContextSize;
                    return false;
                }
                request.dest.WriteCount = config_.ContextSize - request.dest.WriteOffset;
                next_write_offset = config_.ContextSize; // Will break after this one

                if (original_tokens < request.dest.WriteCount) {
                    LOG_ERROR() << "Internal error: Requested write count " << request.dest.WriteCount << " is greater than original tokens " << original_tokens;
                    return 0;
                }

                row->ExpectedNextRemainingCount = original_tokens - request.dest.WriteCount;
                if (row->ExpectedNextRemainingCount < config_.MinDataLength) {
                    row->ExpectedNextRemainingCount = 0;
                }

                if (skipping) {
                    row->SkipRequest = request;
                    row->SkipRequest.dest.WriteOffset = 0; // Will be first write
                    row->SkipRequest.dest.ReadOffset = request.dest.WriteCount;
                    row->SkipRequest.dest.WriteCount = std::min(config_.ContextSize, row->ExpectedNextRemainingCount);
                }
            } else {
                request.dest.WriteCount = original_tokens;
                next_write_offset = write_end + 1;
            }
#if 0
            LOG_INFO() << "FIXME: request = batch_index=" << request.batch_index << " shard_index=" << request.shard_index
                << " shard_span_index=" << request.shard_span_index << " dest.ReadOffset=" << request.dest.ReadOffset
                << " dest.WriteOffset=" << request.dest.WriteOffset << " dest.WriteCount=" << request.dest.WriteCount;
#endif

            if (!skipping) {
                microbatch_requests_.push_back(request);
            }
            ++num_requests;
        }
    }

    uint32_t total_remaining = 0;
    for (auto& row : row_read_results_) {
        total_remaining += row->RemainingCount;

        // If skipping, simulate reading the skipped data
        if (skipping) {
            row->RemainingCount = row->ExpectedNextRemainingCount;
        }
    }

    // Still data available if we are making requests or there is a buffer to read
    return num_requests > 0 || total_remaining > 0;
}

void TokenizedDataLoader::CalculateTotalSteps()
{
    total_steps_ = 0;

    while (GenerateMicrobatchRequests(true)) {
        ++total_steps_;
    }

    // Reset NextSpan() state
    next_shard_index_ = 0;
    for (const auto& shard : shards_) {
        shard->EpochNextSpan = 0;
    }
    for (auto& row : row_read_results_) {
        if (row->RemainingCount != 0) {
            LOG_ERROR() << "Internal error: Row " << row->RemainingCount << " tokens remaining during step calculation";
        }
        row->RemainingCount = 0;
    }
}

void TokenizedDataLoader::Prefill() {
    LOG_DEBUG() << "Prefilling " << config_.MicroBatchSize << "...";

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
#if _DEBUG
    // In debug mode we set all tokens to 0 to make it easier to debug problems
    memset(micro_batch_out, 0, config_.ContextSize * config_.MicroBatchSize * sizeof(uint32_t));
#endif

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

    for (uint32_t batch_index = 0; batch_index < config_.MicroBatchSize; ++batch_index)
    {
        int32_t* output_row = output_batch;
        auto& row = row_read_results_[batch_index];

        *is_continuation = row->RemainingCount > 0;

        const uint32_t written = row->WriteOutput(output_row, config_);
        if (written == 0) {
            continue; // No data to write
        }
        if (row->RemainingCount != row->ExpectedNextRemainingCount) {
            LOG_ERROR() << "Internal error: Batch " << batch_index << " has " << row->RemainingCount
                << " tokens remaining but expected " << row->ExpectedNextRemainingCount;
            return false;
        }

        max_token_count = std::max(max_token_count, written);
        output_row += config_.ContextSize;
        ++num_rows;
        ++is_continuation;
        output_batch += config_.ContextSize;
    }

    // Pad the remaining unwritten rows with padding tokens
    if (num_rows < config_.MicroBatchSize) {
        const uint32_t pad_rows = config_.MicroBatchSize - num_rows;
        for (uint32_t i = 0; i < config_.ContextSize; ++i) {
            output_batch[i] = config_.PaddingToken;
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
        double compression_ratio = 0.0;
        if (total_decompressed_tokens_ > 0) {
            compression_ratio = total_disk_read_ / (double)total_decompressed_tokens_;
        }
        LOG_INFO() << "GetTokenArray: Training data exhausted.  Disk compression: " << compression_ratio << " bytes/token";
        return false;
    }

    Prefill();
    return true;
}
