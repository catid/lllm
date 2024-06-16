# C++ Dataloader

Optimized C++ dataloader for tokenized datasets.

Features:
* All operations are fully pipelined with training and parallelized for speed.
* Uses Zstd and byte planes to efficiently store any n_vocab size (including bytes), saving a ton of disk space.
* Uses an index file to accelerate data lookup.
* Hash to verify entire dataset integrity quickly.
* Supports fast checkpoint resume by skipping ahead a specified number of steps.
* Short strings are concatenated and separated by padding tokens to improve training throughput.


## Example Usage

From Python it looks like this:

```python
from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

loader = DataLoader(data_path)

config = EpochConfig()
config.local_rank = 0
config.local_rank_count = 2
config.padding_token = -1
config.micro_batch_size = 128
config.context_size = 8192
config.min_data_length = 64
config.start_step = 0

loader.begin_epoch(config)

while True:
    batch, is_cont, step, total_steps = loader.get_micro_batch()
    if batch is None:
        print("Dataset exhausted")
        break
```

## Setup

Install the `cpp_dataloader` pip package:

```bash
sudo apt install build-essential cmake

./install.sh
```

Verify that it is working:

```bash
python python test_cpp_dataloader.py
```


## Debugging

There are a few unit tests that can be run to verify the correctness of the code.  To run them, first build the project:

```bash
sudo apt install build-essential cmake

rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j

./test_compressor
./test_tools
./test_worker_pool
./test_mapped_file
./test_uring_file

./test_data_prep
./test_data_loader
```


## Thanks

Latest versions of thirdparty libraries are included:

* `cityhash` from https://github.com/google/cityhash
* `cpppath` from https://github.com/tdegeus/cpppath
* `ryml` from https://github.com/biojppm/rapidyaml
* `liburing` from https://github.com/axboe/liburing
* `zstd` from https://github.com/facebook/zstd
