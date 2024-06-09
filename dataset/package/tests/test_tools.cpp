#include <tools.hpp>

#include <cstring>

bool testAllocateAndFree() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text[] = {1, 2, 3, 4, 5};
    uint32_t text_length = sizeof(tokenized_text);

    // Allocate a buffer
    std::shared_ptr<TokenizedBuffer> buffer = allocator.Allocate(tokenized_text, text_length);

    // Verify the allocated buffer
    if (buffer == nullptr || buffer->Data.size() != text_length ||
        memcmp(buffer->Data.data(), tokenized_text, text_length) != 0) {
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

    LOG_INFO() << "testAllocateAndFree: passed";
    return true;
}

bool testAllocateMultiple() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text1[] = {1, 2, 3};
    uint32_t tokenized_text2[] = {4, 5, 6, 7};
    uint32_t text_length1 = sizeof(tokenized_text1);
    uint32_t text_length2 = sizeof(tokenized_text2);

    // Allocate multiple buffers
    std::shared_ptr<TokenizedBuffer> buffer1 = allocator.Allocate(tokenized_text1, text_length1);
    std::shared_ptr<TokenizedBuffer> buffer2 = allocator.Allocate(tokenized_text2, text_length2);

    // Verify the allocated buffers
    if (buffer1 == nullptr || buffer2 == nullptr ||
        buffer1->Data.size() != text_length1 || buffer2->Data.size() != text_length2 ||
        memcmp(buffer1->Data.data(), tokenized_text1, text_length1) != 0 ||
        memcmp(buffer2->Data.data(), tokenized_text2, text_length2) != 0) {
        return false;
    }

    // Free the buffers
    allocator.Free(buffer1);
    allocator.Free(buffer2);

    LOG_INFO() << "testAllocateMultiple: passed";
    return true;
}

bool testFreeMultiple() {
    TokenizedAllocator allocator;

    // Prepare test data
    uint32_t tokenized_text[] = {1, 2, 3, 4, 5};
    uint32_t text_length = sizeof(tokenized_text);

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

    LOG_INFO() << "testFreeMultiple: passed";
    return true;
}

int main() {
    if (!testAllocateAndFree()) {
        LOG_ERROR() << "testAllocateAndFree failed";
        return -1;
    }

    if (!testAllocateMultiple()) {
        LOG_ERROR() << "testAllocateMultiple failed";
        return -1;
    }

    if (!testFreeMultiple()) {
        LOG_ERROR() << "testFreeMultiple failed";
        return -1;
    }

    LOG_INFO() << "All tests passed";
    return 0;
}
