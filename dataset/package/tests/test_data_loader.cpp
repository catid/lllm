#include <tokenized_data_loader.hpp>

#include "tools.hpp"

bool test_data_verify() {
    if (!VerifyDataset("test_data")) {
        LOG_ERROR() << "test_data_verify test failed";
        return false;
    }

    LOG_INFO() << "test_data_verify test passed";
    return true;
}

bool test_data_loader() {
    TokenizedDataLoader loader;

    uint32_t rank = 0;
    uint32_t local_ranks = 1;

    if (!loader.Start("test_data", rank, local_ranks)) {
        LOG_ERROR() << "Failed to start data loader";
        return false;
    }

    uint64_t seed0 = 0;
    uint64_t seed1 = 0;
    uint32_t k_micro_batch_size = 64;
    uint32_t k_context_size = 8192;

    loader.StartEpoch(seed0, seed1, k_micro_batch_size, k_context_size);

    uint64_t total_spans = 0;

    for (;;) {
        uint32_t micro_batch_size;
        uint32_t num_tokens;
        int32_t output_batch[k_micro_batch_size * k_context_size];
        uint8_t is_continuation[k_micro_batch_size];

        uint64_t t0 = GetNsec();

        if (!loader.GetTokenArray(&micro_batch_size, &num_tokens, output_batch, is_continuation)) {
            break;
        }

        uint64_t t1 = GetNsec();
        double dt_usec = (t1 - t0) / 1000.0;

        total_spans += micro_batch_size;

        LOG_INFO() << "Batch retrieved: micro_batch_size=" << micro_batch_size << ", num_tokens=" << num_tokens << ", dt_usec=" << dt_usec;
        //LOG_INFO() << "Sample data: " << output_batch[0] << " (continuation=" << (int)is_continuation[0] << ")";

        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    LOG_INFO() << "Total spans processed: " << total_spans;

    loader.Stop();

    if (total_spans == 0) {
        LOG_ERROR() << "Data loader did not produce any spans";
        return false;
    }

    LOG_INFO() << "test_data_loader test passed";
    return true;
}

int main() {
    if (!test_data_verify()) {
        return -1;
    }

    if (!test_data_loader()) {
        return -1;
    }

    LOG_INFO() << "All tests passed";

    return 0;
}
