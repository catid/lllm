#include "tokenized_data_verify.hpp"

#include "tools.hpp"
#include "dataloader.hpp"
#include "yaml_parser.hpp"
#include "worker_pool.hpp"

#include <mapped_file.hpp>
#include <city.h>

#include <signal.h>

//------------------------------------------------------------------------------
// Verify

bool verify_files(
    const std::string& index_file_path,
    const std::string& data_file_path,
    uint32_t token_bytes)
{
    MappedFileReader index_reader;
    if (!index_reader.Open(index_file_path)) {
        LOG_INFO() << "Failed to open index file: " << index_file_path;
        return false;
    }

    const char* index_data = reinterpret_cast<const char*>(index_reader.GetData());
    size_t index_size = index_reader.GetSize();
    const char* index_end = index_data + index_size - kIndexEndBytes;
    uint64_t index_data_size = read_uint32_le(index_end);
    uint32_t index_version = static_cast<uint8_t>( index_end[4] );
    uint32_t index_token_bytes = static_cast<uint8_t>( index_end[5] );
    uint64_t index_hash = read_uint64_le(index_end + kIndexEndBytes - 8);
    index_hash ^= CityHash64(index_end, kIndexEndBytes - 8);
    size_t record_count = (index_size - kIndexEndBytes) / kIndexRecordBytes;

    if (DATALOADER_VERSION != index_version) {
        LOG_ERROR() << "Index file version mismatch: expected " << DATALOADER_VERSION << ", found " << index_version;
        return false;
    }
    if (token_bytes != index_token_bytes) {
        LOG_ERROR() << "Index file token_bytes mismatch: expected " << token_bytes << ", found " << index_token_bytes;
        return false;
    }

    MappedFileReader data_reader;
    if (!data_reader.Open(data_file_path)) {
        LOG_INFO() << "Failed to open data file: " << data_file_path;
        return false;
    }

    const char* data_data = reinterpret_cast<const char*>(data_reader.GetData());
    size_t data_size = data_reader.GetSize();
    const char* data_end = data_data + data_size - kDataEndBytes;
    uint32_t data_version = static_cast<uint8_t>( data_end[0] );
    uint32_t data_token_bytes = static_cast<uint8_t>( data_end[1] );
    uint64_t data_hash = read_uint64_le(data_end + kDataEndBytes - 8);
    data_hash ^= CityHash64(data_end, kDataEndBytes - 8);

    if (DATALOADER_VERSION != data_version) {
        LOG_ERROR() << "Data file version mismatch: expected " << DATALOADER_VERSION << ", found " << data_version;
        return false;
    }
    if (token_bytes != data_token_bytes) {
        LOG_ERROR() << "Data file token_bytes mismatch: expected " << token_bytes << ", found " << data_token_bytes;
        return false;
    }
    if (index_data_size + kDataEndBytes != data_size) {
        LOG_ERROR() << "Data file size mismatch: expected " << index_data_size + kDataEndBytes << ", found " << data_size;
        return false;
    }

    for (size_t i = 0; i < record_count; ++i) {
        const char* index_ptr = index_data + i * kIndexRecordBytes;
        uint32_t start = read_uint32_le(index_ptr);
        uint32_t end = read_uint32_le(index_ptr + kIndexRecordBytes);
        //uint32_t original_bytes = read_uint32_le(index_ptr + 4);
        if (end <= start) {
            LOG_INFO() << "Offset end <= start: Entry " << i;
            return false;
        }

        index_hash ^= CityHash64(index_ptr, kIndexRecordBytes);

        uint32_t cbytes = end - start;
        data_hash ^= CityHash64(data_data + start, cbytes);
    }

    if (index_hash != 0) {
        LOG_INFO() << "Index file is corrupted: " << index_file_path;
        return false;
    }

    if (data_hash != 0) {
        LOG_INFO() << "Data file is corrupted: " << data_file_path;
        return false;
    }

    return true;
}

static std::atomic<bool> m_early_stop_flag = ATOMIC_VAR_INIT(false);

bool VerifyDataset(const std::string& data_folder_path)
{
    GlobalIndexYaml global_index_yaml;
    if (!global_index_yaml.Read(data_folder_path)) {
        LOG_INFO() << "Failed to read global index file at " << data_folder_path;
        return false;
    }

    WorkerPool pool;
    pool.Start();

    std::atomic<bool> data_error(false);
    std::atomic<int> files_verified(0);

    const uint64_t t0 = GetNsec();

    m_early_stop_flag = false;

    signal(SIGINT, [](int /*signal*/) {
        m_early_stop_flag = true;
    });

    const int num_files = (int)global_index_yaml.data_files_.size();
    for (int i = 0; i < num_files && !m_early_stop_flag; ++i) {
        const std::string& data_file_path = global_index_yaml.data_files_[i];
        const std::string& index_file_path = global_index_yaml.index_files_[i];

        const uint32_t token_bytes = global_index_yaml.token_bytes_;
        if (global_index_yaml.version_ != DATALOADER_VERSION) {
            LOG_ERROR() << "Global index file version mismatch: expected " << DATALOADER_VERSION << ", found " << global_index_yaml.version_;
            return false;
        }

        const int max_active_tasks = 2;
        pool.QueueTask([t0, token_bytes, &files_verified, num_files, &data_error, data_file_path, index_file_path](int /*worker_index*/) {
            if (data_error) {
                return;
            }
            if (m_early_stop_flag) {
                return;
            }
            if (!verify_files(index_file_path, data_file_path, token_bytes)) {
                data_error = true;
            }
            if (data_error) {
                return;
            }
            const int count = files_verified++;
            if (count % 10 == 1) {
                uint64_t t1 = GetNsec();
                double seconds_elapsed = (t1 - t0) / 1000000000.0;
                double seconds_remaining = seconds_elapsed / count * (num_files - count);
                LOG_INFO() << "Verified " << count << "/" << num_files << " files in "
                    << seconds_elapsed << " seconds (~" << seconds_remaining << " remaining)";
            }
        }, max_active_tasks);

        if (data_error) {
            break;
        }
    }

    if (m_early_stop_flag) {
        LOG_INFO() << "Early stopping due to SIGINT";
        signal(SIGINT, SIG_DFL);
        return true;
    }

    pool.WaitForTasks();
    pool.Stop();

    signal(SIGINT, SIG_DFL);

    if (data_error) {
        LOG_INFO() << "Data verification failed";
        return false;
    }

    uint64_t t1 = GetNsec();
    double seconds_elapsed = (t1 - t0) / 1000000000.0;

    LOG_INFO() << "Verified " << num_files << " files in " << seconds_elapsed << " seconds";
    return true;
}
