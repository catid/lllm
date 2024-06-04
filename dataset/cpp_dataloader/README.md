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

## Thanks

Latest versions of thirdparty libraries are included:

* `cityhash` from https://github.com/google/cityhash
* `cpppath` from https://github.com/tdegeus/cpppath
* `ryml` from https://github.com/biojppm/rapidyaml
* `liburing` from https://github.com/axboe/liburing
* `zstd` from https://github.com/facebook/zstd
