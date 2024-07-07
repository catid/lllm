import ctypes
import os
import site
import torch

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

################################################################################
# FP Compression
################################################################################

lib.fp_compress.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.fp_compress.restype = ctypes.c_void_p

lib.fp_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]
lib.fp_decompress.restype = ctypes.c_void_p

# Python wrapper for fp_compress
def fp_compress(input_tensor, error_bound):
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    if input_tensor.dtype != torch.float32:
        raise TypeError("Input tensor must be of type float32")
    
    input_tensor = input_tensor.contiguous()
    
    result_ptr = lib.fp_compress(input_tensor.data_ptr(), error_bound)
    
    # We need to know the size of the compressed data
    # This information should be provided by the C++ function
    # For now, we'll assume it's the same size as the input
    compressed_size = input_tensor.numel()
    
    # Create a new tensor from the compressed data
    compressed_tensor = torch.from_numpy(
        np.ctypeslib.as_array(ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_uint8)), 
                              shape=(compressed_size,))
    ).to(input_tensor.device)
    
    return compressed_tensor

# Python wrapper for fp_decompress
def fp_decompress(compressed_tensor, nb_ele, cmp_size, error_bound):
    if not isinstance(compressed_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    if compressed_tensor.dtype != torch.uint8:
        raise TypeError("Compressed tensor must be of type uint8")
    
    compressed_tensor = compressed_tensor.contiguous()
    
    result_ptr = lib.fp_decompress(compressed_tensor.data_ptr(), nb_ele, cmp_size, error_bound)
    
    # Create a new tensor from the decompressed data
    decompressed_tensor = torch.from_numpy(
        np.ctypeslib.as_array(ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_float)), 
                              shape=(nb_ele,))
    ).to(compressed_tensor.device)
    
    return decompressed_tensor
