import os
import glob
import argparse
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
from multiprocessing import Process, Queue
import sys
import time
import tiktoken
from cpp_dataloader import DataPreparation
import shutil
import tempfile

def get_parquet_files(directory):
    parquet_files = []
    for root, _, _ in os.walk(directory):
        parquet_files.extend(glob.glob(os.path.join(root, '*.parquet')))
    return parquet_files

def split_array(arr, max_size=4):
    result = []
    i = 0
    while i < len(arr):
        sub_arr = arr[i:min(i + max_size, len(arr))]
        result.append(sub_arr)
        i += len(sub_arr)
    return result

def read_parquet_file(file_paths, args, queue):
    with tempfile.TemporaryDirectory() as temp_dir:
        for file_path in file_paths:
            temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
            shutil.copy(file_path, temp_file_path)

            pfile = pq.ParquetFile(temp_file_path)

            shard_size = (pfile.num_row_groups + args.world_size - 1) // args.world_size
            start_index = args.rank_start * shard_size
            end_index = min(start_index + shard_size, pfile.num_row_groups)

            indices = list(range(start_index, end_index))
            subsets = split_array(indices, max_size=32)

            for group_subset in subsets:
                group = pfile.read_row_groups(row_groups=group_subset, columns=["text"])

                for row in group:
                    text = str(row[0])
                    queue.put(text)

def read_parquet_files(parquet_files, args, queue):
    total_files = len(parquet_files)
    start_time = time.time()
    processed_files = 0

    arrays = split_array(parquet_files, max_size=1)

    for files in arrays:
        process = Process(target=read_parquet_file, args=(files, args, queue))
        process.start()
        process.join()

        processed_files += len(files)
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
