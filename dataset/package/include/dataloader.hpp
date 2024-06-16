/*
    File format constants.
*/

#pragma once

#include <cstdint>

// Format version
#define DATALOADER_VERSION 14

// Main index file
#define DATALOADER_MAIN_INDEX_FILE "index.yaml"

// data_XXX.bin
#define DATALOADER_DATA_FILE_PREFIX "data_"
#define DATALOADER_DATA_FILE_SUFFIX ".bin"

// index_XXX.bin
#define DATALOADER_INDEX_FILE_PREFIX "index_"
#define DATALOADER_INDEX_FILE_SUFFIX ".bin"

/*
    Index record format:
        <compressed data file offset (4 bytes)>
        <original text length in tokens (4 bytes)>

    End of index file format:
        <final data file size (4 bytes)> (must be the first field)
        <version (1 byte)>
        <token_bytes (1 byte)>
        <final index file hash (8 bytes)>

    The hash includes all bytes preceding it.

    To retrieve the size of the compressed data,
    subtract consecutive file offsets.
*/

// Size of a record in the index files
static const int kIndexRecordBytes = 8;

// Size of the end cap record
static const int kIndexEndBytes = 4 + 1 + 1 + 8;

/*
    Data record format:
        <compressed data (4 bytes)>

    End of data file format:
        <version (1 byte)>
        <token_bytes (1 byte)>
        <final data file hash (8 bytes)>

    The hash includes all bytes preceding it.
*/

static const int kDataEndBytes = 1 + 1 + 8;
