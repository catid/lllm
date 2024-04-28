import numpy as np
import argparse
from pathlib import Path
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from . import TRIE_TOKENIZER  # Adjusted the import statement

def main(args):
    # Set up logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Determine the output folder
    output_folder = args.dest if args.dest else Path.home() / 'lllm_dataset'
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Initialize the tokenizer with the vocabulary file
    tokenizer = TRIE_TOKENIZER(args.vocab_file)

    # Load the dataset
    dataset = load_dataset(args.dataset_repo, split=args.split, name=args.dataset_name)

    # Function to tokenize and save a batch of texts
    def tokenize_and_save(texts, filename):
        logger.info(f"Tokenizing {len(texts)} texts")
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

        # Find the max length for padding
        max_length = max(len(chunk) for chunk in chunked_token_ids)
        logger.info(f"Padding to max token length: {max_length}")

        # Pad token ids and convert to numpy array
        padded_tokens = np.array([chunk + [0] * (max_length - len(chunk)) for chunk in chunked_token_ids], dtype=np.int32)

        # Save to memory-mapped file
        file_path = output_folder / filename
        fp = np.memmap(file_path, dtype='uint16', mode='w+', shape=padded_tokens.shape)
        fp[0, 0] = max_length  # Prepend max_length as the first uint16 value
        fp[1:, :] = padded_tokens[:]
        del fp  # Flush changes to disk
        logger.info(f"Saved tokenized data to {file_path}")

    # Process and save the dataset in chunks to manage memory usage
    def process_and_save_dataset(dataset, chunk_size):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(0, len(dataset), chunk_size):
                chunk = dataset[i:i+chunk_size]
                texts = chunk['text']  # Assuming 'text' is the field with text data
                filename = f'tokenized_data_{i//chunk_size}.dat'
                future = executor.submit(tokenize_and_save, texts, filename)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing task: {e}")

    # Run the processing function
    process_and_save_dataset(dataset, args.chunk_size)
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
    args = parser.parse_args()

    main(args)
