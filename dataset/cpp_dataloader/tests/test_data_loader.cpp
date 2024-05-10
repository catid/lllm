#include <tokenized_data_loader.hpp>

#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fstream>

bool test_data_verify() {
    return data_verify("test_data");
}

int main() {
    if (test_data_verify()) {
        std::cout << "test_data_verify test passed" << std::endl;
        return 0;
    } else {
        std::cout << "test_data_verify test failed" << std::endl;
        return -1;
    }
}
