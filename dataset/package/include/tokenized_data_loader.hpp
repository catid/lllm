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
    uint32_t shard_span_index = 0;
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

    void ShuffleIndices(
        uint64_t seed0, uint64_t seed1,
        uint32_t rank,
        uint32_t local_ranks);
    uint64_t GetSpan(uint32_t span_index, uint32_t& bytes_out);

    // The following are filled in by ShuffleIndices():

    // Data for the current epoch.
    uint32_t* EpochSpanData = nullptr;

    // Number of spans in the current epoch.
    uint32_t EpochSpanCount = 0;

    // The next span index to read from the current shard for the current epoch.
    uint32_t EpochNextSpan = 0;

private:
    // We shuffle each shard independently so that multiple threads can
    // work on the shuffling process in parallel.  Each shuffle is O(n).
    std::vector<uint32_t> ShuffledIndices;

    std::shared_ptr<MappedFileReader> IndexFile;

    // Number of spans in the data file.
    // During training we take a subset of the spans for each rank,
    // so the number of spans to do is SpanIndices.size() not NumSpans.
    uint32_t NumSpans = 0;
};


//------------------------------------------------------------------------------
// TokenizedDataLoader

// This is not thread-safe so do not call methods from multiple threads.
class TokenizedDataLoader {
public:
    ~TokenizedDataLoader() {
        Stop();
    }

    // Provide location of the dataset.
    // Also provide the rank of the current process and the number of local ranks.
    // This is used to slice up the dataset for each process running on the node.
    bool Start(
        const std::string& data_folder_path,
        uint32_t rank = 0,
        uint32_t local_ranks = 1);

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
        uint64_t seed0, uint64_t seed1, // random seed for shuffling (synchronzed between nodes)
        uint32_t micro_batch_size, // max size of a microbatch
        uint32_t context_size); // max size of a context

    // Pause until all data from the current microbatch is ready.
    bool WaitUntilDataReady();

    // Get the next microbatch of tokens.
    bool GetTokenArray(
        uint32_t* micro_batch_size, // output: batch size
        uint32_t* num_tokens, // output: number of tokens in the batch
        uint32_t* output_batch, // output: tensor of tokens
        uint8_t* is_continuation); // output: vector of bools, one for each batch

private:
    // The rank of the current process and the number of local ranks
    uint32_t rank_ = 0;
    uint32_t local_ranks_ = 1;

    // Is Stop() called?
    std::atomic<bool> Terminated = ATOMIC_VAR_INIT(false);

    // Global YAML file that has the index and data file paths
    std::shared_ptr<GlobalIndexYaml> global_index_yaml_;

    // One shard for each data file
    std::vector<std::shared_ptr<DataShardContext>> shards_;

    // Order in which to step through the shards.
    // Must be a multiple of the number of shards.
    std::vector<uint32_t> shard_order_;
    uint32_t next_shard_index_ = 0; // [ 0, shard_order_.size() - 1 ]
    static const uint32_t kShardOrderMult = 16;

    std::atomic<uint32_t> shards_ready_ = ATOMIC_VAR_INIT(0);

    // Worker pool for parallelizing various tasks
    WorkerPool pool_;

    std::atomic<bool> worker_error_ = ATOMIC_VAR_INIT(false);

    uint32_t micro_batch_size_ = 0;
    uint32_t context_size_ = 0;
    uint32_t data_stride_ = 0;

    std::atomic<uint32_t> prefill_inflight_ = ATOMIC_VAR_INIT(0);
    std::atomic<bool> prefill_started_ = ATOMIC_VAR_INIT(false);
    std::atomic<bool> prefill_complete_ = ATOMIC_VAR_INIT(false);

    std::atomic<uint64_t> total_disk_read_ = ATOMIC_VAR_INIT(0);
    std::atomic<uint64_t> total_decompressed_bytes_ = ATOMIC_VAR_INIT(0);

    std::mutex output_mutex_;
    std::condition_variable output_condition_;

    std::vector<std::shared_ptr<Decompressor>> decompressors_;

    // If data is longer than GetTokenArray() can return, we break it into
    // several microbatches.  This variable tracks the progress.
    std::vector<uint32_t> output_used_;

    void ResetPrefill();
    void Prefill();

    // Returns false if there is no more data to read.
    bool NextSpan(ReadRequest& request);
};


//------------------------------------------------------------------------------
// Verify

bool VerifyDataset(const std::string& data_folder_path);
