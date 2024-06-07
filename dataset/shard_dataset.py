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
import torch
import yaml

def save_args_to_yaml(args):
    args_file = os.path.join(args.output_dir, "args.yml")
    with open(args_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Arguments saved to {args_file}")

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

def read_parquet_file(file_path, args, queue):
    try:
        pfile = pq.ParquetFile(file_path)

        shard_size = (pfile.num_row_groups + args.world_size - 1) // args.world_size
        start_index = args.rank_start * shard_size
        end_index = min(start_index + shard_size, pfile.num_row_groups)

        indices = list(range(start_index, end_index))
        subsets = split_array(indices, max_size=4)

        for group_subset in subsets:
            groups = pfile.read_row_groups(row_groups=group_subset, columns=["text"])

            for group in groups:
                rows = group.to_pylist()

                for row in rows:
                    queue.put(row)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def read_parquet_files(parquet_files, args, queue):
    total_files = len(parquet_files)
    start_time = time.time()
    processed_files = 0

    arrays = split_array(parquet_files, max_size=3)

    for files in arrays:
        processes = []
        for file_path in files:
            process = Process(target=read_parquet_file, args=(file_path, args, queue))
            process.start()
            processes.append(process)
        for process in processes:
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
    print(f"[Ranks {args.rank_start}-{args.rank_start + args.rank_count - 1}] [{bar}] {elapsed_time_str} / {eta_str} {percent}%")

def tokenizer_worker(queue, output_queue):
    tokenizer = tiktoken.encoding_for_model("gpt-4o")

    while True:
        text = queue.get()
        if text is None:
            output_queue.put(None)
            break

        tokenized_text = tokenizer.encode(text, disallowed_special=())
        output_queue.put(tokenized_text)

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('--dataset_dir', type=str, default="/mnt/Media/datasets/fineweb-edu", help="Path to the Parquet files.")
    parser.add_argument('--rank_start', type=int, default=0, help="First rank for this node.")
    parser.add_argument('--rank_count', type=int, default=1, help="Number of shards for this node.")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of shards.")
    parser.add_argument('--output_dir', type=str, default="dataset_shard", help="Output directory.")

    args = parser.parse_args()

    if args.rank_count != torch.cuda.device_count():
        raise RuntimeError("The --rank_count argument must match the number of GPUs.  Check your `hosts.txt` file configuration.")

    os.makedirs(args.output_dir, exist_ok=True)

    save_args_to_yaml(args)

    exit()

    data_prep = DataPreparation(args.output_dir)

    parquet_files = get_parquet_files(args.dataset_dir)
    queue = Queue(maxsize=128)
    process = Process(target=read_parquet_files, args=(parquet_files, args, queue))
    process.start()

    out_queue = Queue(maxsize=128)

    # Create a pool of tokenizer worker processes
    pool = []
    for _ in range(16):
        p = Process(target=tokenizer_worker, args=(queue, out_queue))
        p.start()
        pool.append(p)

    while True:
        tokenized_text = out_queue.get()
        if tokenized_text is None:  # Check for the sentinel value
            break

        data_prep.write_tokenized_text(tokenized_text)

    # Ensure all worker processes have finished
    for p in pool:
        queue.put(None)
    for p in pool:
        p.join()

    process.join()

    data_prep.destroy()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
