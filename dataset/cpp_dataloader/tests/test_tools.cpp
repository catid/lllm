#include <tools.hpp>

#include <iostream>
#include <cstring>

bool testAllocateAndFree() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text[] = {1, 2, 3, 4, 5};
    uint32_t text_length = sizeof(tokenized_text) / sizeof(uint32_t);

    // Allocate a buffer
    std::shared_ptr<TokenizedBuffer> buffer = allocator.Allocate(tokenized_text, text_length);

    // Verify the allocated buffer
    if (buffer == nullptr || buffer->text.size() != text_length ||
        memcmp(buffer->text.data(), tokenized_text, text_length * sizeof(uint32_t)) != 0) {
        return false;
    }

    // Free the buffer
    allocator.Free(buffer);

    // Allocate another buffer
    std::shared_ptr<TokenizedBuffer> buffer2 = allocator.Allocate(tokenized_text, text_length);

    // Verify that the same buffer is reused
    if (buffer != buffer2) {
        return false;
    }

    // Free the buffer again
    allocator.Free(buffer2);

    return true;
}

bool testAllocateMultiple() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text1[] = {1, 2, 3};
    uint32_t tokenized_text2[] = {4, 5, 6, 7};
    uint32_t text_length1 = sizeof(tokenized_text1) / sizeof(uint32_t);
    uint32_t text_length2 = sizeof(tokenized_text2) / sizeof(uint32_t);

    // Allocate multiple buffers
    std::shared_ptr<TokenizedBuffer> buffer1 = allocator.Allocate(tokenized_text1, text_length1);
    std::shared_ptr<TokenizedBuffer> buffer2 = allocator.Allocate(tokenized_text2, text_length2);

    // Verify the allocated buffers
    if (buffer1 == nullptr || buffer2 == nullptr ||
        buffer1->text.size() != text_length1 || buffer2->text.size() != text_length2 ||
        memcmp(buffer1->text.data(), tokenized_text1, text_length1 * sizeof(uint32_t)) != 0 ||
        memcmp(buffer2->text.data(), tokenized_text2, text_length2 * sizeof(uint32_t)) != 0) {
        return false;
    }

    // Free the buffers
    allocator.Free(buffer1);
    allocator.Free(buffer2);

    return true;
}

bool testFreeMultiple() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text[] = {1, 2, 3, 4, 5};
    uint32_t text_length = sizeof(tokenized_text) / sizeof(uint32_t);

    // Allocate multiple buffers
    std::shared_ptr<TokenizedBuffer> buffer1 = allocator.Allocate(tokenized_text, text_length);
    std::shared_ptr<TokenizedBuffer> buffer2 = allocator.Allocate(tokenized_text, text_length);

    // Free the buffers
    allocator.Free(buffer1);
    allocator.Free(buffer2);

    // Allocate another buffer
    std::shared_ptr<TokenizedBuffer> buffer3 = allocator.Allocate(tokenized_text, text_length);

    // Verify that one of the previously freed buffers is reused
    if (buffer3 != buffer1 && buffer3 != buffer2) {
        return false;
    }

    // Free the buffer
    allocator.Free(buffer3);

    return true;
}

int main() {
    bool testResult1 = testAllocateAndFree();
    bool testResult2 = testAllocateMultiple();
    bool testResult3 = testFreeMultiple();

    if (!testResult1) {
        std::cout << "testAllocateAndFree failed" << std::endl;
        return -1;
    }

    if (!testResult2) {
        std::cout << "testAllocateMultiple failed" << std::endl;
        return -1;
    }

    if (!testResult3) {
        std::cout << "testFreeMultiple failed" << std::endl;
        return -1;
    }

    std::cout << "All tests passed" << std::endl;
    return 0;
}
