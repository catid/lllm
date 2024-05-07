# C++ Dataloader

Optimized C++ dataloader for dataset.

The scripts convert downloaded HuggingFace datasets into this format by tokenizing the text into context-aligned batches and compressing each one into 4 GB files with an index to look up the pieces.  This also supports a hash to check the integrity of the data.

```bash
python setup.py install
```

## Thanks

Third party libraries used:
* `rapidyaml` from https://github.com/biojppm/rapidyaml
* `cpppath` from https://github.com/tdegeus/cpppath
* `cityhash` from https://github.com/google/cityhash
* `zstd` from https://github.com/facebook/zstd
