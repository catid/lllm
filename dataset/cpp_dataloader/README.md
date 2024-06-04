# C++ Dataloader

Optimized C++ dataloader for dataset.

The scripts convert downloaded HuggingFace datasets into this format by tokenizing the text into context-aligned batches and compressing each one into 4 GB files with an index to look up the pieces.  This also supports a hash to check the integrity of the data.

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
#./test_uring_file

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
