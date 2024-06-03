#include <tokenized_data_loader.hpp>

#include "tools.hpp"

bool test_data_verify() {
    if (!data_verify("test_data")) {
        LOG_ERROR() << "test_data_verify test failed";
        return false;
    }

    LOG_INFO() << "test_data_verify test passed";
    return true;
}

bool test_data_loader() {
    TokenizedDataLoader loader;

    if (!loader.Start("test_data")) {
        LOG_ERROR() << "Failed to start data loader";
        return false;
    }

    uint64_t seed0 = 0;
    uint64_t seed1 = 0;
    uint32_t k_micro_batch_size = 16;
    uint32_t k_context_size = 8192;

    loader.StartEpoch(seed0, seed1, k_micro_batch_size, k_context_size);

    for (;;) {
        uint32_t micro_batch_size;
        uint32_t num_tokens;
        uint32_t output_batch[k_micro_batch_size * k_context_size];
        uint8_t is_continuation[k_micro_batch_size];

        if (!loader.GetTokenArray(&micro_batch_size, &num_tokens, output_batch, is_continuation)) {
            break;
        }

        LOG_INFO() << "micro_batch_size=" << micro_batch_size << ", num_tokens=" << num_tokens;
    }

    loader.Stop();

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
