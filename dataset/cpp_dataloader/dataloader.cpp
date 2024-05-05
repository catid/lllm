#include "tokenized_data_loader.hpp"
#include "tokenized_data_prep.hpp"

extern "C" {

//------------------------------------------------------------------------------
// Data Loader

void* data_loader_create(const char* index_file) { 
    TokenizedDataLoader* loader = new TokenizedDataLoader();
    if (!loader->LoadTokenArrays(index_file)) {
        delete loader;
        return nullptr;
    }
    return loader;
}

void data_loader_destroy(void* data_loader) {
    delete static_cast<TokenizedDataLoader*>(data_loader);
}

uint64_t data_loader_start_epoch(
    void* data_loader,
    uint64_t seed0,
    uint64_t seed1,
    uint32_t micro_batch_size,
    uint32_t context_size)
{
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    return loader->StartEpoch(seed0, seed1, micro_batch_size, context_size);
}

bool data_loader_get_micro_batch(
    void* data_loader,
    uint64_t microbatch_index,
    uint32_t context_size,
    uint32_t* micro_batch_size,
    uint32_t* num_tokens,
    uint16_t* output_array)
{
    TokenizedDataLoader* loader = static_cast<TokenizedDataLoader*>(data_loader);
    return loader->GetTokenArray(microbatch_index, context_size, micro_batch_size, num_tokens, output_array);
}


//------------------------------------------------------------------------------
// Data Preparation

void* data_prep_create(const char* data_folder_path) {
    return new TokenizedDataPrep(data_folder_path);
}

void data_prep_destroy(void* data_prep) {
    delete static_cast<TokenizedDataPrep*>(data_prep);
}

bool data_prep_write_tokenized_text(
    void* data_prep,
    const uint16_t* tokenized_text,
    uint32_t text_length)
{
    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    return prep->WriteTokenizedText(tokenized_text, text_length);
}

void data_prep_finalize(void* data_prep) {
    TokenizedDataPrep* prep = static_cast<TokenizedDataPrep*>(data_prep);
    prep->Finalize();
}


//------------------------------------------------------------------------------
// Data Verification

bool data_prep_verify(const char* data_folder_path) {
    return data_verify(data_folder_path);
}

} // extern "C"
