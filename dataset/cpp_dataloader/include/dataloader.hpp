#pragma once

// Main index file
#define DATALOADER_MAIN_INDEX_FILE "index.yaml"

// data_XXX.bin
#define DATALOADER_DATA_FILE_PREFIX "data_"
#define DATALOADER_DATA_FILE_SUFFIX ".bin"

// index_XXX.bin
#define DATALOADER_INDEX_FILE_PREFIX "index_"
#define DATALOADER_INDEX_FILE_SUFFIX ".bin"

static const int kCompressorByteStride = 4; // 32-bit tokens

static const uint32_t kPaddingToken = 0; // <PAD>

extern "C" {


//------------------------------------------------------------------------------
// Data Loader

// Create or destroy a data loader context object
void* data_loader_create(
    const char* index_file,
    uint32_t rank,
    uint32_t local_ranks);
void data_loader_destroy(void* data_loader);

// Start a new epoch
void data_loader_start_epoch(
    void* data_loader,
    uint64_t seed0, uint64_t seed1,
    uint32_t micro_batch_size,
    uint32_t context_size);
bool data_loader_get_micro_batch(
    void* data_loader,
    uint32_t* micro_batch_size,
    uint32_t* num_tokens,
    uint32_t* output_batch,
    uint8_t* is_continuation);


//------------------------------------------------------------------------------
// Data Preparation

void* data_prep_create(const char* data_folder_path);
void data_prep_destroy(void* data_prep);
bool data_prep_write_tokenized_text(
    void* data_prep,
    const uint32_t* tokenized_text,
    uint32_t text_length);

void data_prep_finalize(void* data_prep);


//------------------------------------------------------------------------------
// Data Verification

bool data_verify(const char* data_folder_path);


} // extern "C"
