import ctypes
import numpy as np
import os
import site

# Get the path to site-packages
site_packages_paths = site.getsitepackages()

# Assuming the .so file is in the first site-packages path found
so_file_name = "cpp_dataloader_library.so"
lib_path = None
for path in site_packages_paths:
    potential_path = os.path.join(path, so_file_name)
    print(f"Checking {potential_path}")
    if os.path.exists(potential_path):
        lib_path = potential_path
        break

# Ensure the library path is correct and exists
if not lib_path:
    raise FileNotFoundError(f"Shared library not found. Please read the cpp_dataloader/README.md instructions.")

# Load the shared library
lib = ctypes.CDLL(lib_path)

# Define the C++ function signatures

# Data Loader
lib.data_loader_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
lib.data_loader_create.restype = ctypes.c_void_p

lib.data_loader_destroy.argtypes = [ctypes.c_void_p]
lib.data_loader_destroy.restype = None

lib.data_loader_start_epoch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32]
lib.data_loader_start_epoch.restype = None

lib.data_loader_get_micro_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint8)]
lib.data_loader_get_micro_batch.restype = ctypes.c_bool

# Data Preparation
lib.data_prep_create.argtypes = [ctypes.c_char_p]
lib.data_prep_create.restype = ctypes.c_void_p

lib.data_prep_destroy.argtypes = [ctypes.c_void_p]
lib.data_prep_destroy.restype = None

lib.data_prep_write_tokenized_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
lib.data_prep_write_tokenized_text.restype = ctypes.c_bool

# Data Verification
lib.data_verify.argtypes = [ctypes.c_char_p]
lib.data_verify.restype = ctypes.c_bool

class DataLoader:
    def __init__(self, index_file, rank=0, local_ranks=1):
        self.data_loader = lib.data_loader_create(index_file.encode(), rank, local_ranks)
        if not self.data_loader:
            raise RuntimeError("Failed to create data loader")

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.data_loader:
            lib.data_loader_destroy(self.data_loader)
            self.data_loader = None

    def start_epoch(self, seed0, seed1, micro_batch_size, context_size):
        self.context_size = context_size
        self.microbatch_size = micro_batch_size
        lib.data_loader_start_epoch(self.data_loader, seed0, seed1, micro_batch_size, context_size)

    def get_micro_batch(self):
        micro_batch_size = ctypes.c_uint32()
        num_tokens = ctypes.c_uint32()
        output_array = np.empty((self.microbatch_size, self.context_size), dtype=np.uint32)
        is_continuation = np.empty(self.microbatch_size, dtype=np.uint8)
        success = lib.data_loader_get_micro_batch(self.data_loader,
                                                  ctypes.byref(micro_batch_size),
                                                  ctypes.byref(num_tokens),
                                                  output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                                  is_continuation.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
        if not success or micro_batch_size.value <= 0 or num_tokens.value <= 0:
            return None, None
        return output_array[:micro_batch_size.value, :num_tokens.value], is_continuation[:micro_batch_size.value]

class DataPreparation:
    def __init__(self, data_folder_path):
        self.data_prep = lib.data_prep_create(data_folder_path.encode())
        if not self.data_prep:
            raise RuntimeError("Failed to create data preparation")

    def __del__(self):
        self.destroy()

    def write_tokenized_text(self, tokenized_text):
        tokenized_text = np.asarray(tokenized_text, dtype=np.uint32)
        success = lib.data_prep_write_tokenized_text(self.data_prep, tokenized_text.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(tokenized_text))
        if not success:
            raise RuntimeError("Failed to write tokenized text")

    def destroy(self):
        if self.data_prep:
            lib.data_prep_destroy(self.data_prep)
            self.data_prep = None

class DataVerifier:
    @staticmethod
    def verify(data_folder_path):
        return lib.data_verify(data_folder_path.encode())
