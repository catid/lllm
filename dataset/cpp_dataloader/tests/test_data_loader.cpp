#include <tokenized_data_loader.hpp>

#include "tools.hpp"

bool test_data_verify() {
    return data_verify("test_data");
}

int main() {
    if (!test_data_verify()) {
        LOG_ERROR() << "test_data_verify test failed";
        return -1;
    } else {
        LOG_INFO() << "test_data_verify test passed";
    }

    LOG_INFO() << "All tests passed";

    return 0;
}
