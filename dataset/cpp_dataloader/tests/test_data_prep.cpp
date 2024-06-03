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

static const char* kTestData =
"    // Generate random tokenized text data\r\n" \
"    std::random_device rd;\r\n" \
"    std::mt19937 gen(rd());\r\n" \
"    std::uniform_int_distribution<> dist(0, 255);\r\n" \
"\r\n" \
"    std::vector<std::vector<uint32_t>> tokenized_texts;\r\n" \
"    int num_texts = 100;\r\n" \
"    int max_text_length = 20000;\r\n" \
"\r\n" \
"    for (int i = 0; i < num_texts; ++i) {\r\n" \
"        int text_length = gen() % max_text_length + 1;\r\n" \
"        std::vector<uint32_t> tokenized_text(text_length);\r\n" \
"        for (int j = 0; j < text_length; ++j) {\r\n" \
"            tokenized_text[j] = dist(gen);\r\n" \
"        }\r\n" \
"        tokenized_texts.push_back(tokenized_text);\r\n" \
"    }\r\n";

bool testTokenizedDataPrep() {
    std::string data_folder_path = "test_data";
    TokenizedDataPrep data_prep;
    data_prep.Start(data_folder_path);

    // Generate random tokenized text data
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::vector<uint32_t>> tokenized_texts;
    int num_texts = 100;
    int max_text_length = 20000;
    const int test_data_len = strlen(kTestData);

    for (int i = 0; i < num_texts; ++i) {
        int text_length = gen() % max_text_length + 1;
        std::vector<uint32_t> tokenized_text(text_length);
        for (int j = 0; j < text_length; ++j) {
            tokenized_text[j] = kTestData[j % test_data_len];
        }
        tokenized_texts.push_back(tokenized_text);
    }

    // Write tokenized text data
    for (const auto& tokenized_text : tokenized_texts) {
        if (!data_prep.WriteTokenizedText(tokenized_text.data(), tokenized_text.size())) {
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
