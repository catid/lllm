data_path = "~/lllm/dataset/fineweb-edu"

import random
import time
import pandas as pd
from cpp_dataloader import DataLoader, DataPreparation, DataVerifier
import os
import glob
import argparse

def get_parquet_files(directory):
    """
    Get a list of all Parquet files in the given directory and its subdirectories.

    Args:
    directory (str): The root directory to search for Parquet files.

    Returns:
    list: A list of paths to Parquet files.
    """
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Directory '{folder_path}' created successfully.")
    except OSError as error:
        print(f"Error creating directory '{folder_path}': {error}")

def read_parquet(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    return df

def process_shard(df, args):
    # Split the DataFrame into shards
    shard_size = (len(df) + args.total_shards - 1) // args.total_shards
    start_index = args.shard_index * shard_size
    end_index = start_index + shard_size
    if end_index > len(df):
        end_index = len(df)

    shard = df.iloc[start_index:end_index]

    # Process the shard (in this example, print the lines)
    for line in shard.itertuples(index=False, name=None):
        print(line)

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('dataset_dir', type=str, help="Path to the Parquet files.")
    parser.add_argument('rank_start', type=int, help="First rank for this node.")
    parser.add_argument('rank_count', type=int, help="Number of shards for this node.")
    parser.add_argument('world_size', type=int, help="Total number of shards.")

    args = parser.parse_args()

    parquet_files = get_parquet_files(args.dataset_dir)

    for file_path in parquet_files:
        df = read_parquet(file_path)

        process_shard(df, args)

if __name__ == "__main__":
    main()










# Example usage:
if __name__ == "__main__":
    create_folder_if_not_exists(data_path)

    print("Preparing random tokenized dataset...")
    try:
        start_time = time.time()
        
        prep = DataPreparation(data_path)
        
        for _ in range(4000):
            num_tokens = random.randint(1, 20000)
            tokens = [random.randint(0, 128000) for _ in range(num_tokens)]
            prep.write_tokenized_text(tokens)
        
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
        
        loader = DataLoader(data_path, rank=0, local_ranks=2)
        loader.start_epoch(0, 0, 128, 8192)

        while True:
            batch, is_cont = loader.get_micro_batch()
            if batch is None:
                print("Dataset exhausted")
                break
            print("Batch:", batch)
            print("Is continuation:", is_cont)

        loader.destroy()

        end_time = time.time()
        print(f"DataLoader time: {end_time - start_time:.2f} seconds")
    except RuntimeError as e:
        print(e)
