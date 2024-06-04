# C++ Dataloader

Optimized C++ dataloader for dataset.

The scripts convert downloaded HuggingFace datasets into this format by tokenizing the text into context-aligned batches and compressing each one into 4 GB files with an index to look up the pieces.  This also supports a hash to check the integrity of the data.

## Setup

Build the cpp_dataloader package:

```bash
sudo apt install build-essential cmake

pip install build

rm -rf build dist cpp_dataloader.egg-info && pip uninstall cpp_dataloader -y
python -m build && pip install --force-reinstall dist/*.whl
```

## Thanks

Latest versions of thirdparty libraries are included:

* `cityhash` from https://github.com/google/cityhash
* `cpppath` from https://github.com/tdegeus/cpppath
* `ryml` from https://github.com/biojppm/rapidyaml
* `liburing` from https://github.com/axboe/liburing
* `zstd` from https://github.com/facebook/zstd
