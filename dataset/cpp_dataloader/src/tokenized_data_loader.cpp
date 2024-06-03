#include "tokenized_data_loader.hpp"
#include "tools.hpp"

#include <algorithm>
#include <random>
#include <iostream>

#include <cpppath.h>
#include <ryml.hpp>
#include <mapped_file.hpp>
#include <city.h>


//------------------------------------------------------------------------------
// GlobalIndexYaml

bool GlobalIndexYaml::Read(const std::string& data_folder_path) {
    std::string index_file_path = cpppath::join({data_folder_path, DATALOADER_MAIN_INDEX_FILE});

    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        return false;
    }

    ryml::csubstr yaml_substr((const char*)index_reader.GetData(), index_reader.GetSize());
    ryml::Tree tree = ryml::parse_in_arena(yaml_substr);
    ryml::ConstNodeRef data_files = tree["data_files"];
    ryml::ConstNodeRef index_files = tree["index_files"];
    if (data_files.invalid() || index_files.invalid() ||
        !data_files.is_seq() || !index_files.is_seq() ||
        data_files.num_children() != index_files.num_children()) {
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

void DataShardContext::ShuffleIndices(uint64_t seed0, uint64_t seed1)
{
    std::seed_seq seed{seed0, seed1};
    std::mt19937 rng(seed);

    SpanIndices.reserve(NumSpans);
    uint32_t* span_ptr = SpanIndices.data();
    for (uint32_t i = 0; i < NumSpans; ++i) {
        span_ptr[i] = i;
    }

    std::shuffle(SpanIndices.begin(), SpanIndices.end(), rng);
}

uint64_t DataShardContext::GetSpan(
    uint32_t span_index,
    uint32_t& bytes_out)
{
    uint32_t index = SpanIndices[span_index];
    const uint8_t* span_data = IndexFile->GetData() + index * 4;
    uint32_t start = read_uint32_le(span_data);
    uint32_t end = read_uint32_le(span_data + 4);
    uint32_t span_bytes = end - start;

    bytes_out = span_bytes;
    return start;
}


//------------------------------------------------------------------------------
// TokenizedDataLoader

bool TokenizedDataLoader::Start(const std::string& data_folder_path) {
    Stop();

    global_index_yaml_ = std::make_shared<GlobalIndexYaml>();
    if (!global_index_yaml_->Read(data_folder_path)) {
        LOG_INFO() << "Failed to read global index file at " << data_folder_path;
        return false;
    }

    pool_.Start();

    worker_error_ = false;
    prefill_inflight_ = 0;
    next_shard_index_ = 0;
    shards_ready_ = 0;
    context_size_ = 0;
    micro_batch_size_ = 0;

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

    // Note this waits longer than necessary but this is expected to be a heavy call
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
    LOG_INFO() << "Epoch starting with " << micro_batch_size << " microbatches of " << context_size << " tokens each";

    micro_batch_size_ = micro_batch_size;
    context_size_ = context_size;

    ResetPrefill();

    shards_ready_ = 0;
    for (auto& shard : shards_) {
        pool_.QueueTask([this, shard, seed0, seed1](int /*worker_index*/) {
            shard->ShuffleIndices(seed0, seed1);

            uint32_t ready_count = ++shards_ready_;
            if (ready_count == shards_.size()) {
                Prefill();
            }
        });

        // Use a slightly different seed for each shard.  Not sure if it matters.
        seed0++;
    }
}

void TokenizedDataLoader::ResetPrefill()
{
    std::lock_guard<std::mutex> lock(prefill_mutex_);

    next_shard_index_ = 0;
    prefill_inflight_ = 0;
    prefill_complete_ = false;

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
    LOG_INFO() << "Shuffle complete.  Prefilling " << micro_batch_size_ << "...";

    if (prefill_inflight_ > 0) {
        LOG_ERROR() << "Internal error: Prefill still inflight"; 
        return;
    }

    std::vector<ReadRequest> requests;

    for (uint32_t batch_index = 0; batch_index < micro_batch_size_; ++batch_index)
    {
        auto& decompressor = decompressors_[batch_index];
        const uint32_t decompressed_words = decompressor->Result.size() / 4;
        const uint32_t used = output_used_[batch_index];

        // If there is still data to read, do not prefill this row
        if (used > 0 && used < decompressed_words) {
            continue;
        }

        ReadRequest request;
        request.batch_index = batch_index;
        if (!NextSpan(request)) {
            break; // No more data to read
        }

        output_used_[batch_index] = 0;

        requests.push_back(request);
    }

    if (requests.empty()) {
        LOG_INFO() << "Prepared dataset files contain no more data";
        prefill_complete_ = true;
        return;
    }

    prefill_complete_ = false;
    prefill_inflight_ += (uint32_t)requests.size();

    for (auto& request : requests) {
        pool_.QueueTask([this, request](int /*worker_index*/) {
            // Read offsets from mmap index file
            uint32_t bytes = 0;
            uint64_t offset = shards_[request.shard_index]->GetSpan(request.shard_datum_index, bytes);

            shards_[request.shard_index]->DataFile->Read(offset, bytes,
                [this, request](uint8_t* data, uint32_t bytes)
            {
                auto& decompressor = decompressors_[request.batch_index];
                if (!decompressor->Decompress(data, bytes, kCompressorByteStride)) {
                    LOG_ERROR() << "Failed to decompress data";
                    worker_error_ = true;
                    return;
                }

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
    const uint32_t shard_count = (uint32_t)shards_.size();
    for (uint32_t tries = 0; tries < shard_count; ++tries)
    {
        const uint32_t shard_index = next_shard_index_;

        // Round-robin through the shards
        next_shard_index_ = shard_index + 1;
        if (next_shard_index_ == shard_count) {
            next_shard_index_ = 0;
        }

        auto& shard = shards_[shard_index];
        uint32_t shard_span_index = shard->NextSpan;

        if (shard_span_index < shard->NumSpans) {
            shard->NextSpan = shard_span_index + 1;

            request.shard_index = shard_index;
            request.shard_datum_index = shard_span_index;
            return true;
        }
    }

    return false;
}

bool TokenizedDataLoader::GetTokenArray(
    uint32_t* micro_batch_out,
    uint32_t* max_tokens_out,
    uint32_t* output_batch,
    uint8_t* is_continuation)
{
    // Wait until output is ready
    {
        std::unique_lock<std::mutex> lock(output_mutex_);
        output_condition_.wait(lock, [this]{ return Terminated || prefill_complete_; });
        if (Terminated) {
            return false;
        }
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

        uint32_t used = output_used_[batch_index];
        const uint32_t available = decompressed_words - used;
        if (available <= 0) {
            continue;
        }
        uint32_t copy_bytes = std::min(available, context_size_);

        LOG_INFO() << "batch_index=" << batch_index << ", copy_bytes=" << copy_bytes << ", used=" << used << ", available=" << available << ", decompressed_words=" << decompressed_words;

        memcpy(output_batch, decompressed_ptr + used, copy_bytes * sizeof(uint32_t));
        output_batch += copy_bytes;
        max_token_count = std::max(max_token_count, copy_bytes);

        uint32_t pad_bytes = context_size_ - copy_bytes;
        if (pad_bytes > 0) {
            memset(output_batch, kPaddingToken, pad_bytes * sizeof(uint32_t));
            output_batch += pad_bytes;
        }

        // Signal continuation for this row
        *is_continuation++ = (used != 0);

        output_used_[batch_index] = used + copy_bytes;

        ++num_rows;
    }

    // Pad the remaining unwritten rows with padding tokens
    if (num_rows < micro_batch_size_) {
        uint32_t pad_rows = micro_batch_size_ - num_rows;
        assert(kPaddingToken == 0);
        memset(output_batch, 0, pad_rows * context_size_ * sizeof(uint32_t));
        memset(is_continuation, 0, pad_rows);
    }

    *micro_batch_out = num_rows;
    *max_tokens_out = max_token_count;

    if (num_rows == 0) {
        LOG_INFO() << "GetTokenArray: No data to read";
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

bool data_verify(const std::string& data_folder_path)
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

    const int num_files = (int)global_index_yaml.data_files_.size();
    for (int i = 0; i < num_files; ++i) {
        const std::string& data_file_path = global_index_yaml.data_files_[i];
        const std::string& index_file_path = global_index_yaml.index_files_[i];

        const int max_active_tasks = 2;
        pool.QueueTask([t0, &files_verified, num_files, &data_error, data_file_path, index_file_path](int /*worker_index*/) {
            if (data_error) {
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
                    << seconds_elapsed << " seconds (" << seconds_remaining << " remaining)";
            }
        }, max_active_tasks);

        if (data_error) {
            break;
        }
    }

    pool.WaitForTasks();
    pool.Stop();

    if (data_error) {
        LOG_INFO() << "Data verification failed";
        return false;
    }

    return true;
}
