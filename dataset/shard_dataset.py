data_path = "~/lllm/dataset/fineweb-edu"

import random
import time
import pandas as pd
from package import DataLoader, DataPreparation, DataVerifier
import os
import glob
import argparse

def get_parquet_files(directory):
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def read_parquet(file_path):
    # Read the Parquet file into a DataFrame
    df = pd.read_parquet(file_path)
    return df

def process_shard(data_prep, df, args):
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
        #prep.write_tokenized_text(line.)

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('dataset_dir', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Path to the Parquet files.")
    parser.add_argument('rank_start', type=int, default=0, help="First rank for this node.")
    parser.add_argument('rank_count', type=int, default=1, help="Number of shards for this node.")
    parser.add_argument('world_size', type=int, default=1, help="Total number of shards.")
    parser.add_argument('output_dir', type=str, default="~/lllm/dataset_shard", help="Output directory.")

    args = parser.parse_args()

    parquet_files = get_parquet_files(args.dataset_dir)

    data_prep = DataPreparation(args.output_dir)

    for file_path in parquet_files:
        df = read_parquet(file_path)

        process_shard(data_prep, df, args)

    data_prep.destroy()

if __name__ == "__main__":
    main()
