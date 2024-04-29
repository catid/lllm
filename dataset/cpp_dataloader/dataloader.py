from dataloader import compressData
import ctypes

# Example usage
input_data = b"Hello, World!"
input_size = len(input_data)
output_size = 100  # Provide a sufficient output buffer size

output_buffer = ctypes.create_string_buffer(output_size)
compressed_size = compressData(input_data, input_size, output_buffer, output_size)

compressed_data = output_buffer.raw[:compressed_size]
print("Compressed data:", compressed_data)
