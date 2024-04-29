from dataloader import cpp_write_token_arrays

def write_token_arrays(token_arrays, output_file):
    return cpp_write_token_arrays(token_arrays, len(token_arrays), output_file)
