#pragma once

#include <string>
#include <vector>

//------------------------------------------------------------------------------
// GlobalIndexYaml

struct GlobalIndexYaml {
    bool Read(const std::string& data_folder_path);

    int version_ = 0;
    int token_bytes_ = 0;

    std::vector<std::string> data_files_, index_files_;
};
