import ctypes
import numpy as np
import os
import site
from dataclasses import dataclass

# FIXME
@dataclass
class EpochConfig:
    seed0: int = 0

# Get the path to site-packages
site_packages_paths = site.getsitepackages()

# Assuming the .so file is in the first site-packages path found
so_file_name = "cpp_distributed_library.so"
lib_path = None
for path in site_packages_paths:
    potential_path = os.path.join(path, so_file_name)
    if os.path.exists(potential_path):
        lib_path = potential_path
        break

# Ensure the library path is correct and exists
if not lib_path:
    raise FileNotFoundError(f"Shared library not found. Please read the cpp_distributed/README.md instructions.")

# Load the shared library
lib = ctypes.CDLL(lib_path)

# Define the C++ function signatures

# Data Verification
lib.data_verify.argtypes = [ctypes.c_char_p]
lib.data_verify.restype = ctypes.c_bool

# FIXME
class DataVerifier:
    @staticmethod
    def verify(data_folder_path):
        return lib.data_verify(data_folder_path.encode())
