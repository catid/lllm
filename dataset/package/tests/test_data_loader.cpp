#include <tokenized_data_loader.hpp>

#include "tools.hpp"

#include <cstring>

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

    if (!loader.Start("test_data")) {
        LOG_ERROR() << "Failed to start data loader";
        return false;
    }

    uint32_t k_micro_batch_size = 64;
    uint32_t k_context_size = 8192;
    uint32_t k_start_step = 0;

    EpochConfig config;
    config.LocalRank = 0;
    config.LocalRankCount = 1;
    config.MicroBatchSize = k_micro_batch_size;
    config.ContextSize = k_context_size;
    config.StartStep = k_start_step;

    loader.BeginEpoch(config);

    uint64_t total_spans = 0;
    uint32_t step = 0, total_steps = 0, actual_steps = 0;

    for (;;) {
        uint32_t micro_batch_size;
        uint32_t num_tokens;
        int32_t output_batch[k_micro_batch_size * k_context_size];
        uint8_t is_continuation[k_micro_batch_size];

        uint64_t t0 = GetNsec();

        if (!loader.GetTokenArray(&micro_batch_size, &num_tokens, output_batch, is_continuation, &step, &total_steps)) {
            break;
        }

        uint64_t t1 = GetNsec();
        double dt_usec = (t1 - t0) / 1000.0;

        total_spans += micro_batch_size;
        actual_steps++;

        LOG_INFO() << "Batch retrieved: micro_batch_size=" << micro_batch_size << ", num_tokens=" << num_tokens << ", dt_usec=" << dt_usec;
        //LOG_INFO() << "Sample data: " << output_batch[0] << " (continuation=" << (int)is_continuation[0] << ")";

        std::ostringstream oss;
        for (uint32_t i = 0; i < num_tokens; ++i) {
            if (output_batch[i] > 0) {
                oss << "D";
            } else if (output_batch[i] == 0) {
                LOG_ERROR() << "Found a zero token in the data";
                return false;
                oss << "0";
            } else {
                oss << "_";
            }
        }
        LOG_INFO() << "Sample data:\n" << oss.str();

        //std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    LOG_INFO() << "Total spans processed: " << total_spans;

    loader.Stop();

    if (actual_steps != total_steps) {
        LOG_ERROR() << "Total spans and total steps do not match: " << actual_steps << " vs " << total_steps;
        return false;
    }

    if (total_spans == 0) {
        LOG_ERROR() << "Data loader did not produce any spans";
        return false;
    }

    LOG_INFO() << "test_data_loader test passed";
    return true;
}

bool test_k_start_step(int skip_steps) {
    TokenizedDataLoader loader1, loader2;

    if (!loader1.Start("test_data") || !loader2.Start("test_data")) {
        LOG_ERROR() << "Failed to start data loader";
        return false;
    }

    uint32_t k_micro_batch_size = 8;
    uint32_t k_context_size = 8192;

    EpochConfig config;
    config.LocalRank = 0;
    config.LocalRankCount = 1;
    config.MicroBatchSize = k_micro_batch_size;
    config.ContextSize = k_context_size;

    loader1.BeginEpoch(config);

    // Loader1: Advance N steps
    for (int i = 0; i < skip_steps; ++i) {
        uint32_t micro_batch_size;
        uint32_t num_tokens;
        int32_t output_batch[k_micro_batch_size * k_context_size];
        uint8_t is_continuation[k_micro_batch_size];
        uint32_t step, total_steps;

        if (!loader1.GetTokenArray(&micro_batch_size, &num_tokens, output_batch, is_continuation, &step, &total_steps)) {
            LOG_ERROR() << "Loader1 failed to get token array on step " << i;
            return false;
        }

        //LOG_INFO() << "Loader1 step " << i+1 << ": " << output_batch[0] << " - " << (int)is_continuation[0];
    }

    // Save the output of the final call
    uint32_t micro_batch_size1;
    uint32_t num_tokens1;
    std::vector<int32_t> output_batch1(k_micro_batch_size * k_context_size);
    std::vector<uint8_t> is_continuation1(k_micro_batch_size);
    uint32_t step1, total_steps1;
    if (!loader1.GetTokenArray(&micro_batch_size1, &num_tokens1, output_batch1.data(), is_continuation1.data(), &step1, &total_steps1)) {
        LOG_ERROR() << "Loader1 failed to get token array on the 11th step";
        return false;
    }

    //LOG_INFO() << "Loader1 step next: " << output_batch1[0] << " - " << (int)is_continuation1[0];

    loader1.Stop();

    config.MicroBatchSize = k_micro_batch_size;
    config.ContextSize = k_context_size;
    config.StartStep = skip_steps;

    loader2.BeginEpoch(config);

    uint32_t micro_batch_size2;
    uint32_t num_tokens2;
    std::vector<int32_t> output_batch2(k_micro_batch_size * k_context_size);
    std::vector<uint8_t> is_continuation2(k_micro_batch_size);
    uint32_t step2, total_steps2;
    if (!loader2.GetTokenArray(&micro_batch_size2, &num_tokens2, output_batch2.data(), is_continuation2.data(), &step2, &total_steps2)) {
        LOG_ERROR() << "Loader2 failed to get token array with k_start_step = 10";
        return false;
    }

    //LOG_INFO() << "Loader2 step " << steps << ": " << output_batch2[0] << " - " << (int)is_continuation2[0];

    loader2.Stop();

    // Compare the results
    if (step1 != step2 || total_steps1 != total_steps2) {
        LOG_ERROR() << "Steps and total steps do not match";
        return false;
    }
    if (micro_batch_size1 != micro_batch_size2 || num_tokens1 != num_tokens2 || 
        memcmp(output_batch1.data(), output_batch2.data(), output_batch1.size() * 4) != 0 || 
        memcmp(is_continuation1.data(), is_continuation2.data(), is_continuation1.size()) != 0) {
        LOG_ERROR() << "k_start_step test failed: Outputs do not match";
        return false;
    }

    LOG_INFO() << "k_start_step test passed";
    return true;
}

int main() {
    if (!test_data_verify()) {
        return -1;
    }

    if (!test_data_loader()) {
        return -1;
    }

    for (int i = 0; i < 100; ++i) {
        LOG_INFO() << "---- k_start_step test " << i;
        if (!test_k_start_step(i)) {
            return -1;
        }
    }

    LOG_INFO() << "All tests passed";

    return 0;
}
