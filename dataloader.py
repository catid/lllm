import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Assuming TRIE_TOKENIZER is your custom tokenizer
from model.tokenizer import TRIE_TOKENIZER

class DatasetLoader:
    def __init__(self, dataset_name, dataset_config, tokenizer_file="rwkv_vocab_v20230424.txt"):
        self.tokenizer = TRIE_TOKENIZER(tokenizer_file)
        self.dataset = load_dataset(dataset_name, dataset_config)
        self.tokenized_datasets = self.tokenize_dataset()

    def tokenize_function(self, examples):
        input_ids = []
        for example in examples["text"]:
            encoded = self.tokenizer.encode(example)
            input_ids.append(encoded)
        return {"input_ids": input_ids}

    def tokenize_dataset(self):
        return self.dataset.map(self.tokenize_function, batched=True)

    @staticmethod
    def collate_fn(batch):
        batch_input_ids = [item["input_ids"] for item in batch]
        max_length = max(len(ids) for ids in batch_input_ids)
        batch_input_ids_padded = []
        attention_mask = []

        for ids in batch_input_ids:
            sample_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            attention_mask.append(sample_mask)
            ids_padded = ids + [0] * (max_length - len(ids))
            batch_input_ids_padded.append(ids_padded)

        return {
            "input_ids": torch.tensor(batch_input_ids_padded),
            "attention_mask": torch.tensor(attention_mask)
        }

    def get_dataloader(self, split, batch_size=16):
        return DataLoader(
            self.tokenized_datasets[split],
            shuffle=(split == "train"),
            batch_size=batch_size,
            collate_fn=self.collate_fn
        )
