#include <tokenized_data_prep.hpp>

#include "tools.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fstream>

bool testTokenizedDataPrep() {
    LOG_INFO() << "Writing tokenized data to test_data folder...";

    std::string data_folder_path = "test_data";
    TokenizedDataPrep data_prep;
    const uint32_t token_bytes = 4;
    data_prep.Start(data_folder_path, token_bytes);

    // Generate random tokenized text data
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::vector<uint32_t>> tokenized_texts;
    int num_texts = 400000;
    int max_text_length = 200;

    for (int i = 0; i < num_texts; ++i) {
        int text_length = gen() % max_text_length + 1;

        std::vector<uint32_t> tokenized_text(text_length);
        for (int j = 0; j < text_length; ++j) {
            tokenized_text[j] = (gen() % 128000) + 1;
        }
        tokenized_texts.push_back(tokenized_text);
    }

    // Write tokenized text data
    for (const auto& tokenized_text : tokenized_texts) {
        if (!data_prep.WriteTokens(tokenized_text.data(), tokenized_text.size())) {
            LOG_ERROR() << "Failed to write tokenized text";
            return false;
        }
    }

    // Finalize data preparation
    if (!data_prep.Stop()) {
        LOG_ERROR() << "Failed to finalize data preparation";
        return false;
    }

    data_prep.Stop();

    // Verify the generated files
    std::ifstream data_file("test_data/data_0.bin", std::ios::binary);
    std::ifstream index_file("test_data/index_0.bin", std::ios::binary);

    if (!data_file || !index_file) {
        LOG_ERROR() << "Failed to open generated files";
        return false;
    }

    // TODO: Add more verification of the generated files if needed

    data_file.close();
    index_file.close();

    LOG_INFO() << "Write succeeded. Now run the test_data_loader unit test to verify the data.";

    return true;
}

bool create_directory(const std::string& directory_path) {
#ifdef _WIN32
    if (CreateDirectory(directory_path.c_str(), NULL) != 0) {
        return true;
    }
    else if (GetLastError() == ERROR_ALREADY_EXISTS) {
        return true;
    }
#else
    if (mkdir(directory_path.c_str(), 0777) == 0) {
        return true;
    }
    else if (errno == EEXIST) {
        return true;
    }
#endif
    return false;
}

int main() {
    const std::string directory_path = "test_data";

    if (!create_directory(directory_path)) {
        LOG_ERROR() << "Failed to create test directory";
        return -1;
    }

    if (!testTokenizedDataPrep()) {
        LOG_ERROR() << "TokenizedDataPrep test failed";
        return -1;
    } else {
        LOG_INFO() << "TokenizedDataPrep test passed";
    }

    LOG_INFO() << "All tests passed";

    return 0;
}
