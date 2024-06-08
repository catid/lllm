#include "tokenized_data_loader.hpp"
#include "tools.hpp"

#include <algorithm>
#include <random>
#include <iostream>

#include <cpppath.h>
#include <ryml.hpp>
#include <mapped_file.hpp>
#include <city.h>

#include <signal.h>


//------------------------------------------------------------------------------
// GlobalIndexYaml

bool GlobalIndexYaml::Read(const std::string& data_folder_path) {
    std::string index_file_path = cpppath::join({data_folder_path, DATALOADER_MAIN_INDEX_FILE});

    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        LOG_ERROR() << "Failed to open index file at " << index_file_path;
        return false;
    }

    if (index_reader.GetSize() == 0) {
        LOG_ERROR() << "Index file is empty";
        return false;
    }

    ryml::csubstr yaml_substr((const char*)index_reader.GetData(), index_reader.GetSize());
    ryml::Tree tree = ryml::parse_in_arena(yaml_substr);
    ryml::ConstNodeRef data_files = tree["data_files"];
    ryml::ConstNodeRef index_files = tree["index_files"];
    if (data_files.invalid() || index_files.invalid() ||
        !data_files.is_seq() || !index_files.is_seq() ||
        data_files.num_children() != index_files.num_children()) {
        LOG_ERROR() << "Invalid index file format";
        return false;
    }

    const int num_files = (int)data_files.num_children();
    for (int i = 0; i < num_files; ++i) {
        std::string data_file, index_file;
        ryml::from_chars(data_files[i].val(), &data_file);
        ryml::from_chars(index_files[i].val(), &index_file);

        std::string data_file_path = cpppath::join({data_folder_path, data_file});
        std::string index_file_path = cpppath::join({data_folder_path, index_file});

        data_files_.push_back(data_file_path);
        index_files_.push_back(index_file_path);
    }

    return true;
}


//------------------------------------------------------------------------------
// DataShardContext

bool DataShardContext::Open(
    const std::string& data_file_path,
    const std::string& index_file_path)
{
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

    uint64_t num_spans = (index_file_bytes - 8) / sizeof(uint32_t) - 1;
    if (num_spans > UINT32_MAX) {
        LOG_INFO() << "Too many regions in index file at " << index_file_path;
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
    uint32_t& bytes_out)
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

    const uint8_t* span_data = IndexFile->GetData() + index * 4;
    const uint32_t start = read_uint32_le(span_data);
    const uint32_t end = read_uint32_le(span_data + 4);

    if (start >= end) {
        LOG_ERROR() << "Internal error: Span index " << span_index << " has invalid start=" << start << " end=" << end;
        return 0;
    }

    bytes_out = end - start;
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
            if (!shard->Open(data_file_path, index_file_path)) {
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
            uint64_t offset = shards_[request.shard_index]->GetSpan(
                request.shard_span_index,
                cbytes);

            if (cbytes == 0) {
                LOG_ERROR() << "Invalid span data for shard=" << request.shard_index << " index=" << request.shard_span_index;
                worker_error_ = true;
                return;
            }

            shards_[request.shard_index]->DataFile->Read(offset, cbytes,
                [this, cbytes, offset, request](uint8_t* data, uint32_t bytes)
            {
                if (request.batch_index >= decompressors_.size()) {
                    LOG_ERROR() << "Internal error: Requested batch index " << request.batch_index << " is out of range";
                    worker_error_ = true;
                    return;
                }

                total_disk_read_ += bytes;

                auto& decompressor = decompressors_[request.batch_index];
                if (!decompressor->Decompress(data, bytes, kCompressorByteStride)) {
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


//------------------------------------------------------------------------------
// Verify

bool verify_files(
    const std::string& index_file_path,
    const std::string& data_file_path)
{
    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        LOG_INFO() << "Failed to open index file: " << index_file_path;
        return false;
    }

    const char* index_data = reinterpret_cast<const char*>(index_reader.GetData());
    size_t index_size = index_reader.GetSize();
    uint64_t index_hash = read_uint64_le(index_data + index_size - 8);
    index_hash ^= CityHash64(index_data, sizeof(uint32_t));
    size_t word_count = (index_size - 8) / sizeof(uint32_t);

    MappedFileReader data_reader;
    if (!data_reader.Open(data_file_path)) {
        LOG_INFO() << "Failed to open data file: " << data_file_path;
        return false;
    }

    const char* data_data = reinterpret_cast<const char*>(data_reader.GetData());
    size_t data_size = data_reader.GetSize();
    uint64_t data_hash = read_uint64_le(data_data + data_size - 8);

    for (size_t i = 1; i < word_count; ++i) {
        const char* current_offset_buffer = index_data + i * sizeof(uint32_t);
        uint32_t start = read_uint32_le(current_offset_buffer - 4);
        uint32_t end = read_uint32_le(current_offset_buffer);
        if (end <= start) {
            LOG_INFO() << "Invalid offset: " << current_offset_buffer;
            return false;
        }
        uint32_t bytes = end - start;

        index_hash ^= CityHash64(current_offset_buffer, sizeof(uint32_t));

        data_hash ^= CityHash64(data_data + start, bytes);
    }

    if (index_hash != 0) {
        LOG_INFO() << "Index file is corrupted: " << index_file_path;
        return false;
    }

    if (data_hash != 0) {
        LOG_INFO() << "Data file is corrupted: " << data_file_path;
        return false;
    }

    return true;
}

static std::atomic<bool> m_early_stop_flag = ATOMIC_VAR_INIT(false);

bool VerifyDataset(const std::string& data_folder_path)
{
    GlobalIndexYaml global_index_yaml;
    if (!global_index_yaml.Read(data_folder_path)) {
        LOG_INFO() << "Failed to read global index file at " << data_folder_path;
        return false;
    }

    WorkerPool pool;
    pool.Start();

    std::atomic<bool> data_error(false);
    std::atomic<int> files_verified(0);

    const uint64_t t0 = GetNsec();

    m_early_stop_flag = false;

    signal(SIGINT, [](int /*signal*/) {
        m_early_stop_flag = true;
    });

    const int num_files = (int)global_index_yaml.data_files_.size();
    for (int i = 0; i < num_files && !m_early_stop_flag; ++i) {
        const std::string& data_file_path = global_index_yaml.data_files_[i];
        const std::string& index_file_path = global_index_yaml.index_files_[i];

        const int max_active_tasks = 2;
        pool.QueueTask([t0, &files_verified, num_files, &data_error, data_file_path, index_file_path](int /*worker_index*/) {
            if (data_error) {
                return;
            }
            if (m_early_stop_flag) {
                return;
            }
            if (!verify_files(index_file_path, data_file_path)) {
                data_error = true;
            }
            const int count = files_verified++;
            if (count % 10 == 1) {
                uint64_t t1 = GetNsec();
                double seconds_elapsed = (t1 - t0) / 1000000000.0;
                double seconds_remaining = seconds_elapsed / count * (num_files - count);
                LOG_INFO() << "Verified " << count << "/" << num_files << " files in "
                    << seconds_elapsed << " seconds (~" << seconds_remaining << " remaining)";
            }
        }, max_active_tasks);

        if (data_error) {
            break;
        }
    }

    if (m_early_stop_flag) {
        LOG_INFO() << "Early stopping due to SIGINT";
        signal(SIGINT, SIG_DFL);
        return true;
    }

    pool.WaitForTasks();
    pool.Stop();

    signal(SIGINT, SIG_DFL);

    if (data_error) {
        LOG_INFO() << "Data verification failed";
        return false;
    }

    uint64_t t1 = GetNsec();
    double seconds_elapsed = (t1 - t0) / 1000000000.0;

    LOG_INFO() << "Verified " << num_files << " files in " << seconds_elapsed << " seconds";
    return true;
}
