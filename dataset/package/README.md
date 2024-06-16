# C++ Dataloader

Optimized C++ dataloader for tokenized datasets.

Features:
* All operations are fully pipelined with training for zero delay.
* Compatible with any n_vocab size.
* Uses Zstd with byte planes to save a ton of disk space.
* Negligible memory usage (does not use memory-mapped files).
* Fast random-access disk IO achieved using liburing.
* All operations are full parallelized for speed.
* Uses an index file to accelerate data lookup.
* Hash to verify entire dataset integrity quickly.
* Supports fast checkpoint resume by skipping ahead a specified number of steps without re-reading.
* Short strings are concatenated and separated by padding tokens to improve training throughput.
* Supports seeded random access for reproducibility.

Benchmark results:

When pipelined with training, the dataloader takes approximately 0.01 milliseconds to read each microbatch, so basically it adds no delay to training.

Per 12 CPU cores on an SSD with a (huge) batch of 128 and context size of 8192, you can expect to achieve 6.25 milliseconds read speed per microbatch (measured in Python).


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
