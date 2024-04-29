import numpy as np
import argparse
from pathlib import Path
from datasets import load_dataset
import multiprocessing
import logging
from tqdm import tqdm

from . import TRIE_TOKENIZER  # Adjusted the import statement

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def write_worker(input, output, file_id, stride, output_folder):
    for args in iter(input.get, None):
        try:
            chunked_token_ids = args[0]

            # Find the max length for padding
            max_length = max(len(chunk) for chunk in chunked_token_ids)

            output_dir = Path(output_dir) / ""

            fp = np.memmap(file_path, dtype='uint16', mode='w+', shape=(len(chunked_token_ids), max_length))

            for i, chunk in enumerate(chunked_token_ids):
                # Determine the amount of padding needed
                padding_length = max_length - len(chunk)
                # Create the padded chunk (avoiding creation of a temporary large array)
                padded_chunk = np.array(chunk + [0] * padding_length, dtype=np.uint16)
                # Write directly to the memory-mapped file
                fp[i, :] = padded_chunk

            del fp  # Flush changes to disk

        except Exception as e:
            logger.error(f"token_worker_task exception: {e}")

        file_id += stride
        output.put(result)

def token_worker_task(write_task_queue, tokenizer, texts):
    active_tasks = 0

    try:
        # Tokenize texts
        token_ids = [tokenizer.encode(text) for text in texts]

        # Split each text into chunks of up to 8192 tokens
        chunked_token_ids = []
        for tokens in token_ids:
            chunks = [tokens[i:i+args.context] for i in range(0, len(tokens), args.context)]
            # Drop the last chunk of a set if it has fewer than min_tokens
            if len(chunks) > 0 and len(chunks[-1]) < args.min_tokens:
                chunks = chunks[:-1]
            chunked_token_ids.extend(chunks)

        write_task_queue.put(chunked_token_ids)
        active_tasks += 1

    except Exception as e:
        logger.error(f"token_worker_task exception: {e}")

    return active_tasks

def token_worker(input, output, write_task_queue):
    for args in iter(input.get, None):
        result = token_worker_task(write_task_queue, *args)
        output.put(result)

def produce_token_tasks(tokenizer, dataset, output_folder, args):
    write_task_queue = multiprocessing.Queue()
    write_done_queue = multiprocessing.Queue()

    for i in range(args.num_workers):
        stride = args.num_workers
        multiprocessing.Process(target=write_worker, args=(write_task_queue, write_done_queue, stride, output_folder)).start()

    token_task_queue = multiprocessing.Queue()
    token_done_queue = multiprocessing.Queue()

    for i in range(args.num_workers):
        multiprocessing.Process(target=token_worker, args=(token_task_queue, token_done_queue, write_task_queue)).start()

    active_token_tasks = 0
    active_write_tasks = 0

    for i in tqdm(range(0, len(dataset), args.chunk_size), desc="Producing tasks"):
        chunk = dataset[i:i+args.chunk_size]
        texts = chunk['text']  # Assuming 'text' is the field with text data

        task = (tokenizer, output_folder, texts)

        token_task_queue.put(task)
        active_token_tasks += 1

        # Pause until we have at least one active worker slot
        if active_token_tasks >= args.num_workers:
            active_write_tasks += token_done_queue.get()
            active_token_tasks -= 1

        # Pause until we have at least one active worker slot
        if active_write_tasks >= args.num_workers:
            write_done_queue.get()
            active_write_tasks -= 1

    # Add a sentinel to the queue to signal the end of tasks
    for i in range(args.num_workers):
        token_task_queue.put(None)

    # Wait for all tasks to complete
    while active_token_tasks > 0:
        active_write_tasks += token_done_queue.get()
        active_token_tasks -= 1

    # Add a sentinel to the queue to signal the end of tasks
    # Do this after token tasks are done to avoid race conditions
    for i in range(args.num_workers):
        write_task_queue.put(None)

    # Wait for all tasks to complete
    while active_write_tasks > 0:
        write_done_queue.get()
        active_write_tasks -= 1

def main(args):
    # Determine the output folder
    output_folder = args.dest if args.dest else Path.home() / 'lllm_dataset'
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Initialize the tokenizer with the vocabulary file
    tokenizer = TRIE_TOKENIZER(args.vocab_file)

    # Load the dataset
    dataset = load_dataset(args.dataset_repo, split=args.split, name=args.dataset_name)

    # Run the processing function
    produce_token_tasks(tokenizer, dataset, output_folder, args)
    logger.info("All data has been tokenized and saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize and save dataset.")
    parser.add_argument('--vocab_file', type=str, default="rwkv_vocab_v20230424.txt", help="Path to the vocabulary file")
    parser.add_argument('--dataset_repo', type=str, default="allenai/dolma", help="Name of the dataset repo to load")
    parser.add_argument('--dataset_name', type=str, default="v1_6-sample", help="Which version of dataset to load")
    parser.add_argument('--split', type=str, default="train", help="Dataset split to use")
    parser.add_argument('--chunk_size', type=int, default=1000, help="Number of texts to process in each chunk")
    parser.add_argument('--context', type=int, default=8192, help="Max tokens per chunk")
    parser.add_argument('--min_tokens', type=int, default=1024, help="Min tokens and end of chunk split")
    parser.add_argument('--dest', type=str, default='', help="Destination folder to save the tokenized data. Default is '~/dataset'.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers")
    args = parser.parse_args()

    main(args)
