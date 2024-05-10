#include <tokenized_data_prep.hpp>

#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fstream>

bool testTokenizedDataPrep() {
    std::string data_folder_path = "test_data";
    TokenizedDataPrep data_prep;
    data_prep.Start(data_folder_path);

    // Generate random tokenized text data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 255);

    std::vector<std::vector<uint32_t>> tokenized_texts;
    int num_texts = 100;
    int max_text_length = 1000;

    for (int i = 0; i < num_texts; ++i) {
        int text_length = gen() % max_text_length + 1;
        std::vector<uint32_t> tokenized_text(text_length);
        for (int j = 0; j < text_length; ++j) {
            tokenized_text[j] = dist(gen);
        }
        tokenized_texts.push_back(tokenized_text);
    }

    // Write tokenized text data
    for (const auto& tokenized_text : tokenized_texts) {
        if (!data_prep.WriteTokenizedText(tokenized_text.data(), tokenized_text.size())) {
            std::cout << "Failed to write tokenized text" << std::endl;
            return false;
        }
    }

    // Finalize data preparation
    if (!data_prep.Stop()) {
        std::cout << "Failed to finalize data preparation" << std::endl;
        return false;
    }

    data_prep.Stop();

    // Verify the generated files
    std::ifstream data_file("test_data/data_0.bin", std::ios::binary);
    std::ifstream index_file("test_data/index_0.bin", std::ios::binary);

    if (!data_file || !index_file) {
        std::cout << "Failed to open generated files" << std::endl;
        return false;
    }

    // TODO: Add more verification of the generated files if needed

    data_file.close();
    index_file.close();

    return true;
}

int main() {
    if (testTokenizedDataPrep()) {
        std::cout << "TokenizedDataPrep test passed" << std::endl;
        return 0;
    } else {
        std::cout << "TokenizedDataPrep test failed" << std::endl;
        return 1;
    }
}
