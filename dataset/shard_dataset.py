import random
import time
import pandas as pd
import os
import glob
import argparse
import tiktoken
from multiprocessing import Process, Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from cpp_dataloader import DataLoader, DataPreparation, DataVerifier
import sys

def get_parquet_files(directory):
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df

def process_shard(data_prep, df, args, tokenizer):
    shard_size = (len(df) + args.world_size - 1) // args.world_size
    start_index = args.rank_start * shard_size
    end_index = min(start_index + shard_size, len(df))

    print(f"Processing {start_index} to {end_index} of {len(df)} pieces...")

    shard = df.iloc[start_index:end_index]

    total = 0

    for line in shard.itertuples(index=False, name=None):
        text = line[0]

        total += 1
        print(f"Processing {len(text)}...{total}")

        tokenized_text = tokenizer.encode(text)
        data_prep.write_tokenized_text(tokenized_text)

        print(f"Processed {len(text)}.")

def print_progress_bar(args, iteration, total, start_time, bar_length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    elapsed_time = time.time() - start_time
    eta = (elapsed_time / iteration) * (total - iteration) if iteration > 0 else 0
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    bar_filled = int(bar_length * iteration // total)
    bar = '#' * bar_filled + '-' * (bar_length - bar_filled)
    sys.stdout.write(f"\r[Ranks {args.rank_start}-{args.rank_start + args.rank_count - 1}] {percent}% [{bar}] {elapsed_time_str} / {eta_str}")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('--dataset_dir', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Path to the Parquet files.")
    parser.add_argument('--rank_start', type=int, default=0, help="First rank for this node.")
    parser.add_argument('--rank_count', type=int, default=1, help="Number of shards for this node.")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of shards.")
    parser.add_argument('--output_dir', type=str, default="dataset_shard", help="Output directory.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing shards {args.rank_start} to {args.rank_start + args.rank_count - 1} of {args.world_size}.")

    tokenizer = tiktoken.encoding_for_model("gpt-4")

    parquet_files = get_parquet_files(args.dataset_dir)

    data_prep = DataPreparation(args.output_dir)

    total_files = len(parquet_files)
    start_time = time.time()

    for i, file_path in enumerate(parquet_files):
        df = read_parquet(file_path)

        p = Process(target=process_shard, args=(data_prep, df, args, tokenizer))
        p.start()
        p.join()

        print_progress_bar(args, i + 1, total_files, start_time)

    data_prep.destroy()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
