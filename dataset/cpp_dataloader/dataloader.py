import ctypes
import numpy as np
from ctypes.util import find_library

class DataLoader:
    def __init__(self):
        lib_path = find_library("cpp_dataloader")
        if lib_path is None:
            raise FileNotFoundError("Could not find the dataloader library.")
        self.lib = ctypes.CDLL(lib_path)
        self.lib.WriteTokenArrays.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_char_p]
        self.lib.WriteTokenArrays.restype = ctypes.c_bool

    def write_token_arrays(self, token_arrays, output_file):
        array_ptrs = (ctypes.POINTER(ctypes.c_uint32) * len(token_arrays))()
        array_sizes = (ctypes.c_size_t * len(token_arrays))()

        for i, array in enumerate(token_arrays):
            array_ptrs[i] = array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
            array_sizes[i] = len(array)

        output_file_cstr = output_file.encode('utf-8')
        success = self.lib.WriteTokenArrays(array_ptrs, array_sizes, len(token_arrays), output_file_cstr)
        return success

def write_token_arrays(token_arrays, output_file):
    dataloader = DataLoader()
    return dataloader.write_token_arrays(token_arrays, output_file)

# Example usage
if __name__ == "__main__":
    # Example token arrays
    token_arrays = [
        np.array([1, 2, 3, 4, 5], dtype=np.uint32),
        np.array([6, 7, 8, 9, 10], dtype=np.uint32),
        np.array([11, 12, 13, 14, 15], dtype=np.uint32)
    ]

    output_file = "token_data.bin"
    success = write_token_arrays(token_arrays, output_file)
    if success:
        print(f"Token arrays written to {output_file} successfully.")
    else:
        print("Failed to write token arrays to disk.")