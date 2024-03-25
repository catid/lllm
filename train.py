# Set the environment variables
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

import deepspeed
from deepspeed import comm

# Initialize DeepSpeed
deepspeed.init_distributed()
rank = deepspeed.comm.get_rank()
world_size = deepspeed.comm.get_world_size()

# Load the dataset
dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")
print(dataset)

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

comm.barrier()

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"])

split_dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

tokenized_dataset = split_dataset.map(tokenize_function, batched=True, num_proc=16, remove_columns=["text"])

# Define the training arguments with DeepSpeed configuration
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    fp16=True,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    deepspeed="deepspeed_config.json",  # Path to DeepSpeed configuration file
)

# Create the Trainer with DeepSpeed
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model with DeepSpeed
trainer.train()
