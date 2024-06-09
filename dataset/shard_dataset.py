import os
import glob
import argparse
import pyarrow.parquet as pq
import requests
import shutil
import git
from multiprocessing import Process, Queue, Pool
import time
import tempfile
import yaml
import tiktoken
from tqdm import tqdm
from cpp_dataloader import DataPreparation
import torch

def save_args_to_yaml(args, additional_keys=None):
    args_dict = vars(args)
    
    if additional_keys:
        args_dict.update(additional_keys)
    
    args_file = os.path.join(args.output_dir, "args.yml")
    
    with open(args_file, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)
    
    print(f"Arguments saved to {args_file}")

def split_array(arr, max_size=4):
    result = []
    i = 0
    while i < len(arr):
        sub_arr = arr[i:min(i + max_size, len(arr))]
        result.append(sub_arr)
        i += len(sub_arr)
    return result

def read_parquet_file(file_path, queue):
    try:
        #print(f"Reading {file_path}...")

        pfile = pq.ParquetFile(file_path)

        indices = list(range(0, pfile.num_row_groups))
        subsets = split_array(indices, max_size=4)

        for group_subset in subsets:
            groups = pfile.read_row_groups(row_groups=group_subset, columns=["text"])

            for group in groups:
                rows = group.to_pylist()

                for row in rows:
                    queue.put(row)

        temp_dir = os.path.dirname(file_path)
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        queue.put(None)

def read_parquet_files(total_files, download_queue, queue):
    processed_files = 0
    start_time = time.time()

    while True:
        file_path = download_queue.get()
        if file_path is None:
            break

        read_parquet_file(file_path, queue)
        processed_files += 1
        print_progress_bar(processed_files, total_files, start_time)

    queue.put(None)

def print_progress_bar(iteration, total, start_time, bar_length=40):
    percent = f"{100 * (iteration / float(total)):.1f}"
    elapsed_time = time.time() - start_time
    eta = (elapsed_time / iteration) * (total - iteration) if iteration > 0 else 0
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
    bar_filled = int(bar_length * iteration // total)
    bar = '#' * bar_filled + '-' * (bar_length - bar_filled)
    print(f"[{bar}] {elapsed_time_str} / {eta_str} {percent}%")

def tokenizer_worker(encoding, queue, output_queue):
    tokenizer = tiktoken.get_encoding(encoding)

    while True:
        text = queue.get()
        if text is None:
            output_queue.put(None)
            break

        tokenized_text = tokenizer.encode(text, disallowed_special=())
        output_queue.put(tokenized_text)

def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def get_file_list(dataset_user, dataset_name):
    HTTP_URL = f"https://huggingface.co/api/datasets/{dataset_user}/{dataset_name}"
    response = requests.get(HTTP_URL)
    files = []

    if response.status_code == 200:
        data = response.json()
        for item in data["siblings"]:
            file = item["rfilename"]
            if file.startswith("data/"):
                files.append(file)
    else:
        print(f"Error fetching data: {response.status_code}")

    return files

def clone_and_move_file(file_path, temp_dir, dataset_user, dataset_name):
    start_time = time.time()

    dst_file_path = os.path.join(temp_dir, file_path)
    if os.path.exists(dst_file_path):
        print(f"File {dst_file_path} already exists. Skipping...")
        return dst_file_path
    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

    # Initialize the Git repository
    repo = git.Repo.init(temp_dir)

    # Enable sparse-checkout
    sparse_checkout_path = os.path.join(temp_dir, '.git', 'info', 'sparse-checkout')
    with open(sparse_checkout_path, 'w') as f:
        f.write(file_path + '\n')

    # Set the config for sparse checkout
    repo.git.config('core.sparseCheckout', 'true')

    # Add the remote repository
    REPO_URL = f"https://huggingface.co/datasets/{dataset_user}/{dataset_name}.git"
    origin = repo.create_remote('origin', REPO_URL)

    # Fetch and checkout the specific file
    origin.fetch()
    repo.git.checkout('main')

    # Move the file to the output directory
    src_file_path = os.path.join(temp_dir, file_path)
    shutil.move(src_file_path, dst_file_path)

    # Calculate and print the download speed
    end_time = time.time()
    file_size = os.path.getsize(dst_file_path)  # File size in bytes
    download_time = end_time - start_time  # Time in seconds
    download_speed = file_size / download_time / (1000 * 1000) # Speed in MB/s

    print(f"\nDownloaded {dst_file_path}: {download_speed:.2f} MB/s ({file_size/1000000.0:.2f} MB)")
    return dst_file_path

def download_worker(filename_queue, download_queue, args):
    curdir = os.curdir

    while True:
        file_path = filename_queue.get()
        if file_path is None:
            break

        temp_dir = tempfile.mkdtemp()
        #print(f"Using temporary directory for download: {temp_dir}")

        os.chdir(temp_dir)

        dst_file_path = clone_and_move_file(file_path, temp_dir, args.dataset_user, args.dataset_name)

        os.chdir(curdir)

        download_queue.put(dst_file_path)

def main():
    parser = argparse.ArgumentParser(description="Read and process shards of a Parquet file.")
    parser.add_argument('--dataset_user', type=str, default="HuggingFaceFW", help="Dataset user.")
    parser.add_argument('--dataset_name', type=str, default="fineweb-edu", help="Dataset name.")
    parser.add_argument('--rank_start', type=int, default=0, help="First rank for this node.")
    parser.add_argument('--rank_count', type=int, default=1, help="Number of shards for this node.")
    parser.add_argument('--world_size', type=int, default=1, help="Total number of shards.")
    parser.add_argument('--output_dir', type=str, default="dataset_shard", help="Output directory.")
    parser.add_argument('--encoding', type=str, default="o200k_base", help="Tiktoken encoding.")
    parser.add_argument("--just_args", action="store_true", help="Just write the args file and exit.")
    parser.add_argument("--byte_tokens", action="store_true", help="Tokenize using byte tokens instead of word tokens.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of worker processes.")

    args = parser.parse_args()

    if args.rank_count != torch.cuda.device_count():
        raise RuntimeError("Your --rank_count parameter must match the number of GPUs. Check your `hosts.txt` file configuration.")

    delete_folder_contents(args.output_dir)

    # Write some extra keys to the args file
    tokenizer = tiktoken.get_encoding(args.encoding)
    n_vocab = tokenizer.max_token_value + 1

    extra_keys = {
        "n_vocab": n_vocab
    }

    save_args_to_yaml(args, extra_keys)

    if args.just_args:
        print("Just wrote the args file. Exiting.")
        return

    data_prep = DataPreparation(args.output_dir, byte_tokens=args.byte_tokens)

    # Get the list of files to download
    file_paths = get_file_list(args.dataset_user, args.dataset_name)

    # Determine the shard of files to download
    total_files = len(file_paths)
    shard_size = (total_files + args.world_size - 1) // args.world_size
    start_index = args.rank_start * shard_size
    end_index = min(start_index + args.rank_count, total_files)
    shard_file_paths = file_paths[start_index:end_index]
    shard_file_count = len(shard_file_paths)

    print(f"\nDownloading {len(shard_file_paths)} files (shard {start_index}-{end_index})...")

    filename_queue = Queue()
    download_queue = Queue(maxsize=2)

    for file_path in shard_file_paths:
        filename_queue.put(file_path)

    # Create a pool of download worker processes
    download_pool = []
    for _ in range(2):
        p = Process(target=download_worker, args=(filename_queue, download_queue, args))
        p.start()
        download_pool.append(p)

    queue = Queue(maxsize=128)
    process = Process(target=read_parquet_files, args=(shard_file_count, download_queue, queue))
    process.start()

    out_queue = Queue(maxsize=128)

    # Create a pool of tokenizer worker processes
    pool = []
    for _ in range(16):
        p = Process(target=tokenizer_worker, args=(args.encoding, queue, out_queue))
        p.start()
        pool.append(p)

    while True:
        tokenized_text = out_queue.get()
        if tokenized_text is None:  # Check for the sentinel value
            break

        data_prep.write_tokens(tokenized_text)

    print(f"\nWaiting for download processes to finish...\n")

    # Ensure all worker processes have finished
    for p in download_pool:
        filename_queue.put(None)
    for p in download_pool:
        p.join()

    print(f"\nWaiting for tokenizer processes to finish...\n")

    # Ensure all worker processes have finished
    for p in pool:
        queue.put(None)
    for p in pool:
        p.join()

    print(f"\nWaiting for parquet reader to finish...\n")

    process.join()

    print(f"\nWaiting for dataset preparation to finish...\n")

    data_prep.destroy()

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
