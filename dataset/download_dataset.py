# Modify configuration:
DATASET_USER = "HuggingFaceFW" # EDIT THIS
DATASET_NAME = "fineweb-edu" # EDIT THIS
OUTPUT_DIR = "/mnt/Media/datasets/fineweb-edu" # EDIT THIS

DOWNLOAD_TEMP_DIR = "download_temp"
NUM_WORKERS = 4

import requests
import os
import shutil
import git
from tqdm import tqdm
import time
import multiprocessing
import tempfile

# Function to get the list of files to download
def get_file_list():
    HTTP_URL = f"https://huggingface.co/api/datasets/{DATASET_USER}/{DATASET_NAME}"
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

# Function to clone the repository and move the file
def clone_and_move_file(file_path, temp_dir, out_dir):
    start_time = time.time()

    dst_file_path = os.path.join(out_dir, file_path)
    if os.path.exists(dst_file_path):
        print(f"File {dst_file_path} already exists. Skipping...")
        return
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
    REPO_URL = f"https://huggingface.co/datasets/{DATASET_USER}/{DATASET_NAME}.git"
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
    download_speed = file_size / download_time / (1024 * 1024)  # Speed in MB/s

    print(f"\nDownloaded {dst_file_path}: {download_speed:.2f} MB/s")

# Worker function for multiprocessing
def worker(args):
    base_temp_dir, out_dir, file_path = args
    temp_dir = tempfile.mkdtemp(dir=base_temp_dir)

    os.chdir(temp_dir)

    clone_and_move_file(file_path, temp_dir, out_dir)

    os.chdir(base_temp_dir)

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    temp_dir = os.path.abspath(DOWNLOAD_TEMP_DIR)
    out_dir = os.path.abspath(OUTPUT_DIR)

    # Ensure the output directories exist
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=False)

    # Get the list of files to download
    file_paths = get_file_list()

    print(f"\nDownloading {len(file_paths)} files...")

    worker_args = [(temp_dir, out_dir, file_path) for file_path in file_paths]

    # Create a progress bar
    with tqdm(total=len(file_paths), desc="Downloading files") as pbar:
        # Start multiprocessing pool to download files in parallel
        with multiprocessing.Pool(NUM_WORKERS) as pool:
            for _ in pool.imap_unordered(worker, worker_args):
                pbar.update(1)
