#include "tokenized_data_loader.hpp"
#include "tokenized_data_prep.hpp"

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <Python.h>

extern "C" {

//------------------------------------------------------------------------------
// Data Loader

void* data_loader_create(
    const char* index_file,
    uint32_t rank,
    uint32_t local_ranks)
{
    TokenizedDataLoader* loader = new TokenizedDataLoader();
    if (!loader->Start(index_file, rank, local_ranks)) {
        delete loader;
        return nullptr;
    }
    return loader;
}

void data_loader_destroy(void* data_loader) {
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    if (!loader) {
        return;
    }
    loader->Stop();
    delete loader;
}

void data_loader_start_epoch(
    void* data_loader,
    uint64_t seed0, uint64_t seed1,
    uint32_t micro_batch_size,
    uint32_t context_size,
    uint32_t start_step)
{
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    loader->StartEpoch(seed0, seed1, micro_batch_size, context_size, start_step);
}

bool data_loader_get_micro_batch(
    void* data_loader,
    uint32_t* micro_batch_size,
    uint32_t* num_tokens,
    int32_t* output_batch,
    uint8_t* is_continuation)
{
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    return loader->GetTokenArray(micro_batch_size, num_tokens, output_batch, is_continuation);
}


//------------------------------------------------------------------------------
// Data Preparation

void* data_prep_create(const char* data_folder_path, uint32_t token_bytes) {
    TokenizedDataPrep* prep = new TokenizedDataPrep();
    prep->Start(data_folder_path, token_bytes);
    return prep;
}

void data_prep_destroy(void* data_prep) {
    delete static_cast<TokenizedDataPrep*>(data_prep);
}

bool data_prep_write_tokens(
    void* data_prep,
    const uint32_t* tokenized_text,
    uint32_t text_length)
{
    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    return prep->WriteTokens(tokenized_text, text_length);
}

bool data_prep_write_bytes(
    void* data_prep,
    const uint8_t* text,
    uint32_t bytes)
{
    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    return prep->WriteTokens(text, bytes);
}


//------------------------------------------------------------------------------
// Data Verification

bool data_verify(const char* data_folder_path) {
    return VerifyDataset(data_folder_path);
}

} // extern "C"
