import numpy as np
from datasets import load_dataset
from model.tokenizer import TRIE_TOKENIZER

# Define the path to the vocabulary file and tokenizer
vocab_file = "rwkv_vocab_v20230424.txt"
tokenizer = TRIE_TOKENIZER(vocab_file)

# Load the dataset
dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")

# Function to tokenize and save a batch of texts
def tokenize_and_save(texts, filename):
    # Tokenize texts
    token_ids = [tokenizer.encode(text) for text in texts]
    
    # Find the max length for padding
    max_length = max(len(tokens) for tokens in token_ids)
    
    # Pad token ids and convert to numpy array
    padded_tokens = np.array([tokens + [0] * (max_length - len(tokens)) for tokens in token_ids], dtype=np.int32)
    
    # Save to memory-mapped file
    fp = np.memmap(filename, dtype='int32', mode='w+', shape=padded_tokens.shape)
    fp[:] = padded_tokens[:]
    del fp  # Flush changes to disk

# Process and save the dataset in chunks to manage memory usage
def process_and_save_dataset(dataset, chunk_size=1000):
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        texts = chunk['text']  # Assuming 'text' is the field with text data
        filename = f'tokenized_data_{i//chunk_size}.dat'
        tokenize_and_save(texts, filename)
        print(f'Saved chunk {i//chunk_size} to {filename}')

# Run the processing function
process_and_save_dataset(dataset)

print("All data has been tokenized and saved.")
