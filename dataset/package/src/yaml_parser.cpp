#include "yaml_parser.hpp"

#include "dataloader.hpp"
#include "tools.hpp"
#include "mapped_file.hpp"

#include <cpppath.h>
#include <ryml.hpp>

//------------------------------------------------------------------------------
// GlobalIndexYaml

bool GlobalIndexYaml::Read(const std::string& data_folder_path) {
    std::string index_file_path = cpppath::join({data_folder_path, DATALOADER_MAIN_INDEX_FILE});

    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        LOG_ERROR() << "Failed to open index file at " << index_file_path;
        return false;
    }

    if (index_reader.GetSize() == 0) {
        LOG_ERROR() << "Index file is empty";
        return false;
    }

    ryml::csubstr yaml_substr((const char*)index_reader.GetData(), index_reader.GetSize());
    ryml::Tree tree = ryml::parse_in_arena(yaml_substr);
    ryml::ConstNodeRef data_files = tree["data_files"];
    ryml::ConstNodeRef index_files = tree["index_files"];
    ryml::ConstNodeRef version = tree["version"];
    ryml::ConstNodeRef token_bytes = tree["token_bytes"];
    if (data_files.invalid() || index_files.invalid() ||
        !data_files.is_seq() || !index_files.is_seq() ||
        data_files.num_children() != index_files.num_children() ||
        version.invalid() || token_bytes.invalid() ||
        !version.is_keyval() || !token_bytes.is_keyval()) {
        LOG_ERROR() << "Invalid index file format";
        return false;
    }

    ryml::from_chars(version.val(), &version_);
    ryml::from_chars(token_bytes.val(), &token_bytes_);

    const int num_files = (int)data_files.num_children();
    for (int i = 0; i < num_files; ++i) {
        std::string data_file, index_file;
        ryml::from_chars(data_files[i].val(), &data_file);
        ryml::from_chars(index_files[i].val(), &index_file);

        std::string data_file_path = cpppath::join({data_folder_path, data_file});
        std::string index_file_path = cpppath::join({data_folder_path, index_file});

        data_files_.push_back(data_file_path);
        index_files_.push_back(index_file_path);
    }

    return true;
}
