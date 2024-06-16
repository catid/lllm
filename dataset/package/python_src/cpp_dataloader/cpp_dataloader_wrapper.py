import ctypes
import numpy as np
import os
import site
from dataclasses import dataclass

@dataclass
class EpochConfig:
    seed0: int = 0
    seed1: int = 0
    local_rank: int = 0
    local_rank_count: int = 1
    padding_token: int = -1
    micro_batch_size: int = 4
    context_size: int = 4096
    min_data_length: int = 64
    start_step: int = 0

# Get the path to site-packages
site_packages_paths = site.getsitepackages()

# Assuming the .so file is in the first site-packages path found
so_file_name = "cpp_dataloader_library.so"
lib_path = None
for path in site_packages_paths:
    potential_path = os.path.join(path, so_file_name)
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
lib.data_loader_create.argtypes = [ctypes.c_char_p]
lib.data_loader_create.restype = ctypes.c_void_p

lib.data_loader_destroy.argtypes = [ctypes.c_void_p]
lib.data_loader_destroy.restype = None

class EpochConfigCpp(ctypes.Structure):
    _fields_ = [
        ("seed0", ctypes.c_uint64),
        ("seed1", ctypes.c_uint64),
        ("local_rank", ctypes.c_uint32),
        ("local_rank_count", ctypes.c_uint32),
        ("padding_token", ctypes.c_int32),
        ("micro_batch_size", ctypes.c_uint32),
        ("context_size", ctypes.c_uint32),
        ("min_data_length", ctypes.c_uint32),
        ("start_step", ctypes.c_uint32),
    ]

lib.data_loader_begin_epoch.argtypes = [ctypes.c_void_p, ctypes.POINTER(EpochConfigCpp)]
lib.data_loader_begin_epoch.restype = None

lib.data_loader_get_micro_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
lib.data_loader_get_micro_batch.restype = ctypes.c_bool

# Data Preparation
lib.data_prep_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.data_prep_create.restype = ctypes.c_void_p

lib.data_prep_destroy.argtypes = [ctypes.c_void_p]
lib.data_prep_destroy.restype = None

lib.data_prep_write_tokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
lib.data_prep_write_tokens.restype = ctypes.c_bool

# Data Verification
lib.data_verify.argtypes = [ctypes.c_char_p]
lib.data_verify.restype = ctypes.c_bool

class DataLoader:
    def __init__(self, index_file):
        self.data_loader = lib.data_loader_create(index_file.encode())
        if not self.data_loader:
            raise RuntimeError("Failed to create data loader")

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.data_loader:
            lib.data_loader_destroy(self.data_loader)
            self.data_loader = None

    def begin_epoch(self, config):
        self.context_size = config.context_size
        self.microbatch_size = config.micro_batch_size
        epoch_config = EpochConfigCpp(
            seed0=config.seed0,
            seed1=config.seed1,
            local_rank=config.local_rank,
            local_rank_count=config.local_rank_count,
            padding_token=config.padding_token,
            micro_batch_size=config.micro_batch_size,
            context_size=config.context_size,
            min_data_length=config.min_data_length,
            start_step=config.start_step,
        )
        lib.data_loader_begin_epoch(self.data_loader, ctypes.byref(epoch_config))

    def get_micro_batch(self):
        micro_batch_size = ctypes.c_uint32()
        num_tokens = ctypes.c_uint32()
        steps = ctypes.c_uint32()
        total_steps = ctypes.c_uint32()
        output_array = np.empty((self.microbatch_size, self.context_size), dtype=np.int32)
        is_continuation = np.empty(self.microbatch_size, dtype=np.uint8)
        success = lib.data_loader_get_micro_batch(self.data_loader,
                                                  ctypes.byref(micro_batch_size),
                                                  ctypes.byref(num_tokens),
                                                  output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                                  is_continuation.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                                  ctypes.byref(steps),
                                                  ctypes.byref(total_steps))
        if not success or micro_batch_size.value <= 0 or num_tokens.value <= 0:
            return None, None, None, None
        return output_array[:micro_batch_size.value, :num_tokens.value], is_continuation[:micro_batch_size.value], steps.value, total_steps.value

class DataPreparation:
    def __init__(self, data_folder_path, byte_tokens):
        self.token_bytes = 1 if byte_tokens else 4
        self.data_prep = lib.data_prep_create(data_folder_path.encode(), self.token_bytes)
        if not self.data_prep:
            raise RuntimeError("Failed to create data preparation")

    def __del__(self):
        self.destroy()

    def write_tokens(self, tokens):
        if self.token_bytes == 1:
            raise RuntimeError("DataPreparation: Tokens must be written in byte format. Use write_bytes()")
        tokens = np.asarray(tokens, dtype=np.uint32)
        success = lib.data_prep_write_tokens(
            self.data_prep,
            tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            len(tokens))
        if not success:
            raise RuntimeError("Failed to write tokenized text")

    def write_bytes(self, byte_data):
        if self.token_bytes != 1:
            raise RuntimeError("DataPreparation: Bytes must be written in word format. Use write_tokens()")
        byte_data = np.asarray(byte_data, dtype=np.uint8)
        success = lib.data_prep_write_tokens(
            self.data_prep,
            byte_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(byte_data))
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
