#pragma once

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
#include "yaml_parser.hpp"

//------------------------------------------------------------------------------
// EpochConfig

struct EpochConfig {
    // Random seed for shuffling (synchronzed between nodes)
    uint64_t Seed0 = 0;
    uint64_t Seed1 = 0;

    // Local rank of the current process and the number of local ranks
    uint32_t LocalRank = 0;
    uint32_t LocalRankCount = 1;

    // Padding token
    int32_t PaddingToken = -1;

    // Max size of a microbatch
    uint32_t MicroBatchSize = 4;

    // Max size of a context
    uint32_t ContextSize = 4096;

    // If data is shorter than this, we discard it.
    // If data is longer than context then the remaining tokens may be discarded
    // if they are under this limit.
    uint32_t MinDataLength = 64;

    // Start step for the epoch
    uint32_t StartStep = 0;
};


//------------------------------------------------------------------------------
// ReadRequest

struct BufferTarget {
    // How the data should be written to the output buffer by WriteOutput()
    uint32_t ReadOffset = 0;
    uint32_t WriteOffset = 0;
    uint32_t WriteCount = 0;
};

struct ReadRequest {
    // Which batch row this will go into
    uint32_t batch_index = 0;

    // Which shard (data file) to read from
    uint32_t shard_index = 0;

    // Index of the span of compressed data to read
    uint32_t shard_span_index = 0;

    // Output buffer operation
    BufferTarget dest;
};


//------------------------------------------------------------------------------
// RowReadResults

struct ReadResult {
    std::shared_ptr<Decompressor> Decomp;

    BufferTarget Dest;
};

class RowReadResults {
public:
    // Queued up set of decompressors for this batch row
    std::vector<ReadResult> Results;

    // Number of tokens remaining for the first read result.
    // This can be longer than the context size.
    // This includes ready data, so it is updated on WriteOutput().
    uint32_t RemainingCount = 0;

    // Written by GenerateMicrobatchRequests(), this is the number of tokens
    // we expect to have left over after WriteOutput() is called.
    uint32_t ExpectedNextRemainingCount = 0;

    // Next IO request that carries over from the previous step,
    // used for skipping over data before the first output.
    ReadRequest SkipRequest;

    std::shared_ptr<Decompressor> AddResult(const BufferTarget& dest);
    void Reset();

    // Writes to output and leaves behind any partial data
    uint32_t WriteOutput(int32_t* output_row, const EpochConfig& config);

protected:
    std::mutex Lock;
    std::vector<std::shared_ptr<Decompressor>> Freed;
    std::vector<ReadResult> NextResults;
};


//------------------------------------------------------------------------------
// DataShardContext

struct DataShardContext {
    ~DataShardContext() {
        Close();
    }

    bool Open(
        const std::string& data_file_path,
        const std::string& index_file_path,
        uint32_t token_bytes);
    void Close();

    std::shared_ptr<AsyncUringReader> DataFile;

    void ShuffleIndices(
        uint64_t seed0, uint64_t seed1,
        uint32_t rank,
        uint32_t local_ranks);
    bool GetSpan(
        uint32_t span_index,
        uint64_t& offset_out,
        uint32_t& cbytes_out,
        uint32_t& original_tokens_out);

    // The following are filled in by ShuffleIndices():

    // Data for the current epoch.
    uint32_t* EpochSpanData = nullptr;

    // Number of spans in the current epoch.
    uint32_t EpochSpanCount = 0;

    // The next span index to read from the current shard for the current epoch.
    uint32_t EpochNextSpan = 0;

private:
    uint32_t token_bytes_ = 0;

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

        start_step is used to resume training from a checkpoint.
        One step = One GetTokenArray() call.

        We concatenate short strings together so that there are fewer than
        max_end_padding padding characters at the end of each batch row.
        This allows us to process more tokens per batch.
        More info here: https://github.com/Dao-AILab/flash-attention/issues/432
    */
    void BeginEpoch(const EpochConfig& config);

    // Pause until all data from the current microbatch is ready.
    bool WaitUntilDataReady();

    // Get the next microbatch of tokens.
    // For byte models this will still be 32-bit tokens ranging from 0..255.
    // In all cases, the value of -1 will be used for padding.
    bool GetTokenArray(
        uint32_t* micro_batch_size, // output: batch size
        uint32_t* num_tokens, // output: number of tokens in the batch
        int32_t* output_batch, // output: tensor of tokens
        uint8_t* is_continuation, // output: vector of bools, one for each batch
        uint32_t* step, // output: current step number
        uint32_t* total_steps); // output: total number of steps in dataset

private:
    uint32_t token_bytes_ = 0;

    EpochConfig config_;

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

    std::atomic<uint32_t> prefill_inflight_ = ATOMIC_VAR_INIT(0);
    std::atomic<bool> prefill_started_ = ATOMIC_VAR_INIT(false);
    std::atomic<bool> prefill_complete_ = ATOMIC_VAR_INIT(false);

    std::atomic<uint64_t> total_disk_read_ = ATOMIC_VAR_INIT(0);
    std::atomic<uint64_t> total_decompressed_tokens_ = ATOMIC_VAR_INIT(0);

    std::mutex output_mutex_;
    std::condition_variable output_condition_;

    // List of IO requests for the current microbatch
    std::vector<ReadRequest> microbatch_requests_;

    std::vector<std::shared_ptr<RowReadResults>> row_read_results_;

    uint32_t current_step_ = 0;
    uint32_t total_steps_ = 0;

    void ResetPrefill();
    void Prefill();

    // Fill microbatch_requests_ with requests for the current microbatch.
    // Returns true if data is available to read.
    bool GenerateMicrobatchRequests(bool skipping);

    // Returns false if there is no more data to read.
    bool NextSpan(ReadRequest& request);

    void PostRequests();

    void CalculateTotalSteps();
};


//------------------------------------------------------------------------------
// Verify

bool VerifyDataset(const std::string& data_folder_path);
