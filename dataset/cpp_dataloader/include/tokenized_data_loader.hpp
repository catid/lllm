/*
    FIXME:
    What happens if the next epoch is started and we still have prefill inflight?
    How to prefetch the next epoch?
    How to wait until data is available?
*/

#pragma once

/*
    Possible data loading options:

    (1) Always run from first to last token in the dataset.

        This is the default behavior for most training scripts.
        It can be efficiently implemented using mmap.

    (2) Load from random spans in the dataset. [We are here.]

        This is the current implementation.
        Instead of using mmap we skip the page cache and read
        directly from disk using io_uring and IO_DIRECT.
        It may not have any advantage over sequential access.

        The spans written to the data file must match the context
        size requested during training or else a random subset
        will be selected, which wastes training tokens.

    (3) Select a subset of the dataset to load.

        This is the ideal implementation, which requires more work.
        We need to nail down the selection criterion.
        But in any case the access pattern will be random.
        This is an important optimization for training.
*/

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <random>

#include "dataloader.hpp"
#include "compressor.hpp"
#include "mapped_file.hpp"
#include "uring_file.hpp"
#include "worker_pool.hpp"


//------------------------------------------------------------------------------
// ReadRequest

struct ReadRequest {
    uint32_t batch_index = 0;
    uint32_t shard_index = 0;
    uint32_t shard_datum_index = 0;
};


//------------------------------------------------------------------------------
// GlobalIndexYaml

struct GlobalIndexYaml {
    bool Read(const std::string& data_folder_path);

    std::vector<std::string> data_files_, index_files_;
};


//------------------------------------------------------------------------------
// DataShardContext

struct DataShardContext {
    ~DataShardContext() {
        Close();
    }

    bool Open(
        const std::string& data_file_path,
        const std::string& index_file_path);
    void Close();

    std::shared_ptr<AsyncUringReader> DataFile;
    std::shared_ptr<MappedFileReader> IndexFile;
    uint32_t NumSpans = 0;

    void ShuffleIndices(uint64_t seed0, uint64_t seed1);
    uint64_t GetSpan(uint32_t span_index, uint32_t& bytes_out);

    // We shuffle each shard independently so that multiple threads can
    // work on the shuffling process in parallel.  Each shuffle is O(n).
    std::vector<uint32_t> SpanIndices;
};


//------------------------------------------------------------------------------
// TokenizedDataLoader

class TokenizedDataLoader {
public:
    ~TokenizedDataLoader() {
        Stop();
    }

    bool Start(const std::string& data_folder_path);

    // This stops all background work without waiting for it to complete.
    void Stop();

    /*
        Each datum is accessed once, but in a new random order each epoch.
        We perform round-robin selection of data shards.
        Each data shard access order is shuffled.
        This means that shorter data shards are completed first, so it
        is not fully uniformly selecting each datum.  But our end goal is
        to implement RHO-loss, so this is fine for now.
    */
    void StartEpoch(
        uint64_t seed0,
        uint64_t seed1,
        uint32_t micro_batch_size,
        uint32_t context_size,
        uint32_t data_stride);

    // Get the next microbatch of tokens.
    bool GetTokenArray(
        uint32_t* micro_batch_size,
        uint32_t* num_tokens,
        uint32_t* output_batch,
        uint8_t* is_continuation);

private:
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);
    std::shared_ptr<GlobalIndexYaml> global_index_yaml_;

    std::vector<std::shared_ptr<DataShardContext>> shards_;

    WorkerPool pool_;

    std::atomic<bool> worker_error_ = ATOMIC_VAR_INIT(false);

    uint32_t micro_batch_size_ = 0;
    uint32_t context_size_ = 0;
    uint32_t data_stride_ = 0;

    std::atomic<uint32_t> shards_ready_ = ATOMIC_VAR_INIT(0);

    std::mutex prefill_mutex_;
    uint32_t next_shard_index_ = 0;
    std::vector<uint32_t> shard_next_datum_;
    std::atomic<uint32_t> prefill_inflight_ = ATOMIC_VAR_INIT(0);

    std::mutex output_mutex_;
    std::condition_variable output_condition_;

    std::vector<std::shared_ptr<Decompressor>> decompressors_;

    // If data is longer than GetTokenArray() can return, we break it into
    // several microbatches.  This variable tracks the progress.
    std::vector<uint32_t> output_used_;

    void ResetPrefill();
    void Prefill();
};


//------------------------------------------------------------------------------
// Verify

bool data_verify(const std::string& data_folder_path);
