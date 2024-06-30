# Distributed C++ Extensions

Provides a wrapper around transmitting compressed floating point data between servers via sockets.  Uses `quiche` for HTTP3 transport security and BBR congestion control.  Uses `cuSZp` for CUDA-accelerated floating point compression.

# Setup

```bash
git submodule update --init --recursive
sudo apt install cmake build-essential cargo

./install.sh
```
