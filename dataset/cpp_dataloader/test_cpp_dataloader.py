import random
import time
from cpp_dataloader import DataLoader, DataPreparation, DataVerifier
import os

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Directory '{folder_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{folder_path}': {error}")

data_path = "python_test_data"

# Example usage:
if __name__ == "__main__":
    create_folder_if_not_exists(data_path)

    print("Preparing random tokenized dataset (~30 seconds)...")
    try:
        start_time = time.time()
        
        prep = DataPreparation(data_path)
        
        # Add random tokens from 0..128000 for 1-20000 tokens per write, and at least 1000 of these
        for _ in range(10000):
            num_tokens = random.randint(1, 20000)
            tokens = [random.randint(0, 128000) for _ in range(num_tokens)]
            prep.write_tokenized_text(tokens)
        
        prep.finalize()
        
        end_time = time.time()
        print(f"DataPreparation time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)

    print("Example usage of DataVerifier")
    try:
        start_time = time.time()
        
        is_valid = DataVerifier.verify(data_path)

        if not is_valid:
            raise RuntimeError("Data verification failed")
        
        end_time = time.time()
        print(f"Successfully verified in {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)

    print("Example usage of DataLoader")
    try:
        start_time = time.time()
        
        loader = DataLoader(data_path, rank=0, local_ranks=2)
        loader.start_epoch(0, 0, 32, 8192)
        batch, is_cont = loader.get_micro_batch()
        print("Batch:", batch)
        print("Is continuation:", is_cont)

        end_time = time.time()
        print(f"DataLoader time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)
