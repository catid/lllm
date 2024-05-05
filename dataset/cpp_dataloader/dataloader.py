import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL("path/to/your/shared/library.so")

# Define the C++ function signatures
lib.data_loader_create.argtypes = [ctypes.c_char_p]
lib.data_loader_create.restype = ctypes.c_void_p

lib.data_loader_destroy.argtypes = [ctypes.c_void_p]
lib.data_loader_destroy.restype = None

lib.data_loader_start_epoch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32]
lib.data_loader_start_epoch.restype = ctypes.c_uint64

lib.data_loader_get_micro_batch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint16)]
lib.data_loader_get_micro_batch.restype = ctypes.c_bool

lib.data_prep_create.argtypes = [ctypes.c_char_p]
lib.data_prep_create.restype = ctypes.c_void_p

lib.data_prep_destroy.argtypes = [ctypes.c_void_p]
lib.data_prep_destroy.restype = None

lib.data_prep_write_tokenized_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32]
lib.data_prep_write_tokenized_text.restype = ctypes.c_bool

lib.data_prep_finalize.argtypes = [ctypes.c_void_p]
lib.data_prep_finalize.restype = None

class DataLoader:
    def __init__(self, index_file):
        self.data_loader = lib.data_loader_create(index_file.encode())
        if not self.data_loader:
            raise RuntimeError("Failed to create data loader")

    def __del__(self):
        lib.data_loader_destroy(self.data_loader)

    def start_epoch(self, seed0, seed1, micro_batch_size, context_size):
        return lib.data_loader_start_epoch(self.data_loader, seed0, seed1, micro_batch_size, context_size)

    def get_micro_batch(self, microbatch_index, context_size):
        micro_batch_size = ctypes.c_uint32()
        num_tokens = ctypes.c_uint32()
        output_array = np.empty(context_size, dtype=np.uint16)
        success = lib.data_loader_get_micro_batch(self.data_loader, microbatch_index, context_size,
                                                  ctypes.byref(micro_batch_size), ctypes.byref(num_tokens),
                                                  output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))
        if not success:
            raise RuntimeError("Failed to get micro batch")
        return output_array[:num_tokens.value].reshape(-1, micro_batch_size.value)

class DataPreparation:
    def __init__(self, data_folder_path):
        self.data_prep = lib.data_prep_create(data_folder_path.encode())
        if not self.data_prep:
            raise RuntimeError("Failed to create data preparation")

    def __del__(self):
        lib.data_prep_destroy(self.data_prep)

    def write_tokenized_text(self, tokenized_text):
        tokenized_text = np.asarray(tokenized_text, dtype=np.uint16)
        success = lib.data_prep_write_tokenized_text(self.data_prep, tokenized_text.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), len(tokenized_text))
        if not success:
            raise RuntimeError("Failed to write tokenized text")

    def finalize(self):
        lib.data_prep_finalize(self.data_prep)
