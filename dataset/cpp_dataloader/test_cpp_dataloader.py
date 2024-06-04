import random
import time
from cpp_dataloader import DataLoader, DataPreparation, DataVerifier

# Example usage:
if __name__ == "__main__":
    # Example usage of DataPreparation
    try:
        start_time = time.time()
        
        prep = DataPreparation("test_data")
        
        # Add random tokens from 0..128000 for 1-20000 tokens per write, and at least 1000 of these
        for _ in range(1000):
            num_tokens = random.randint(1, 20000)
            tokens = [random.randint(0, 128000) for _ in range(num_tokens)]
            prep.write_tokenized_text(tokens)
        
        prep.finalize()
        
        end_time = time.time()
        print(f"DataPreparation time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)

    # Example usage of DataVerifier
    try:
        start_time = time.time()
        
        is_valid = DataVerifier.verify("test_data")
        print("Data verification result:", is_valid)
        
        end_time = time.time()
        print(f"DataVerification time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)

    # Example usage of DataLoader
    try:
        start_time = time.time()
        
        loader = DataLoader("test_data", rank=0, local_ranks=1)
        loader.start_epoch(0, 0, 32, 8192)
        batch, is_cont = loader.get_micro_batch()
        print("Batch:", batch)
        print("Is continuation:", is_cont)
        
        end_time = time.time()
        print(f"DataLoader time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)
