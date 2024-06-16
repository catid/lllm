import random
import time
from cpp_dataloader import DataLoader, DataPreparation, DataVerifier, EpochConfig
import os

import numpy as np
np.set_printoptions(threshold=np.inf)

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

    print("Preparing random tokenized dataset...")
    try:
        start_time = time.time()
        
        prep = DataPreparation(data_path, byte_tokens=False)
        
        for _ in range(8000):
            num_tokens = random.randint(1, 10000)
            tokens = [random.randint(0, 128000) for _ in range(num_tokens)]
            prep.write_tokens(tokens)

        prep.destroy()
        
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

        loader = DataLoader(data_path)

        config = EpochConfig()
        config.local_rank = 0
        config.local_rank_count = 2
        config.padding_token = -1
        config.micro_batch_size = 128
        config.context_size = 8192
        config.min_data_length = 64
        config.start_step = 0

        loader.begin_epoch(config)

        while True:
            batch, is_cont, step, total_steps = loader.get_micro_batch()
            if batch is None:
                print("Dataset exhausted")
                break
            print("Batch:", batch.shape)
            print("Is continuation:", is_cont, ", Step:", step, ", Total steps:", total_steps)

        loader.destroy()

        end_time = time.time()
        print(f"DataLoader time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)
