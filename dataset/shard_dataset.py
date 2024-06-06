import random
import time
import pandas as pd
from package import DataLoader, DataPreparation, DataVerifier
import os
import glob
import argparse
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_parquet_files(directory):
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def read_parquet(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    return df

def tokenize_and_write(data_prep, tokenizer, text):
    tokenized_text = tokenizer.encode(text)
    data_prep.write_tokenized_text(tokenized_text)

def process_shard(data_prep, df, args, tokenizer):
    # Split the DataFrame into shards
    shard_size = (len(df) + args.world_size - 1) // args.world_size
    start_index = args.rank_start * shard_size
    end_index = start_index + shard_size
    if end_index > len(df):
        end_index = len(df)

    shard = df.iloc[start_index:end_index]

    with ThreadPoolExecutor(max_workers=16) as executor:  # Limit to 16 threads
        futures = []
        for line in shard.itertuples(index=False, name=None):
            text = " ".join(map(str, line))  # Convert the tuple to a single string
            futures.append(executor.submit(tokenize_and_write, data_prep, tokenizer, text))

        for future in as_completed(futures):
            future.result()  # Raise exceptions if any

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('--dataset_dir', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Path to the Parquet files.")
    parser.add_argument('--rank_start', type=int, default=0, help="First rank for this node.")
    parser.add_argument('--rank_count', type=int, default=1, help="Number of shards for this node.")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of shards.")
    parser.add_argument('--output_dir', type=str, default="dataset_shard", help="Output directory.")

    args = parser.parse_args()

    # Initialize the tokenizer
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    parquet_files = get_parquet_files(args.dataset_dir)

    data_prep = DataPreparation(args.output_dir)

    for file_path in tqdm(parquet_files, desc="Processing Parquet files"):
        df = read_parquet(file_path)
        process_shard(data_prep, df, args, tokenizer)

    data_prep.destroy()

if __name__ == "__main__":
    main()
