//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "tokenized_data_loader.hpp"
#include "tokenized_data_prep.hpp"

//#include <Python.h>

extern "C" {

//------------------------------------------------------------------------------
// Data Loader

void* data_loader_create(const char* index_file, uint32_t rank, uint32_t local_ranks) {
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

void data_loader_start_epoch(void* data_loader, uint64_t seed0, uint64_t seed1, uint32_t micro_batch_size, uint32_t context_size) {
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    loader->StartEpoch(seed0, seed1, micro_batch_size, context_size);
}

bool data_loader_get_micro_batch(void* data_loader, uint32_t* micro_batch_size, uint32_t* num_tokens, uint32_t* output_batch, uint8_t* is_continuation) {
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    return loader->GetTokenArray(micro_batch_size, num_tokens, output_batch, is_continuation);
}

//------------------------------------------------------------------------------
// Data Preparation

void* data_prep_create(const char* data_folder_path) {
    TokenizedDataPrep* prep = new TokenizedDataPrep();
    prep->Start(data_folder_path);
    return prep;
}

void data_prep_destroy(void* data_prep) {
    delete static_cast<TokenizedDataPrep*>(data_prep);
}

bool data_prep_write_tokenized_text(void* data_prep, const uint32_t* tokenized_text, uint32_t text_length) {
    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    return prep->WriteTokenizedText(tokenized_text, text_length);
}

//------------------------------------------------------------------------------
// Data Verification

bool data_verify(const char* data_folder_path) {
    return VerifyDataset(data_folder_path);
}

} // extern "C"
