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

// Often times models are not trained with a padding token (OpenAI for example).
// We use a special padding token to represent padding after the last token.
static const int32_t kPaddingToken = -1; // <PAD>
