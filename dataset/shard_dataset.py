import os
import glob
import argparse
import pandas as pd
from multiprocessing import Process, Queue
import sys
import time
import tiktoken
from cpp_dataloader import DataPreparation

def get_parquet_files(directory):
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def read_parquet_file(file_path, args, queue):
    df = pd.read_parquet(file_path)

    shard_size = (len(df) + args.world_size - 1) // args.world_size
    start_index = args.rank_start * shard_size
    end_index = min(start_index + shard_size, len(df))

    shard = df.iloc[start_index:end_index]

    for line in shard.itertuples(index=False, name=None):
        text = line[0]

        queue.put(text)

def read_parquet_files(parquet_files, args, queue):
    total_files = len(parquet_files)
    start_time = time.time()
    processed_files = 0

    # Iterate in pairs
    for i in range(0, len(parquet_files), 2):
        if i + 1 < len(parquet_files):
            # Process a pair of files
            file_path1 = parquet_files[i]
            file_path2 = parquet_files[i + 1]

            process1 = Process(target=read_parquet_file, args=(file_path1, args, queue))
            process1.start()
            process2 = Process(target=read_parquet_file, args=(file_path2, args, queue))
            process2.start()

            process1.join()
            processed_files += 1
            print_progress_bar(args, processed_files, total_files, start_time)

            process2.join()
            processed_files += 1
            print_progress_bar(args, processed_files, total_files, start_time)

        else:
            # Process the last file if the count is odd
            file_path = parquet_files[i]

            process = Process(target=read_parquet_file, args=(file_path, args, queue))
            process.start()
            process.join()

            processed_files += 1
            print_progress_bar(args, processed_files, total_files, start_time)

    queue.put(None)  # Sentinel to indicate completion

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

    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    data_prep = DataPreparation(args.output_dir)

    parquet_files = get_parquet_files(args.dataset_dir)
    queue = Queue(maxsize=128)
    process = Process(target=read_parquet_files, args=(parquet_files, args, queue))
    process.start()

    while True:
        text = queue.get()
        if text is None:  # Check for the sentinel value
            break

        tokenized_text = tokenizer.encode(text)
        data_prep.write_tokenized_text(tokenized_text)

    process.join()
    data_prep.destroy()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
