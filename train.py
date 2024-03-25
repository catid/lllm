# Set the environment variables
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

import torch
import numpy as np
import random

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

# Initialize DeepSpeed
deepspeed.init_distributed()
rank = comm.get_rank() # global rank
local_rank = comm.get_local_rank()
world_size = comm.get_world_size()

# Deepspeed logging functions
def log_0(msg):
    log_dist(msg, ranks=[0])
def log_all(msg):
    log_dist(msg, ranks=[-1])

def is_main_process():
    return comm.get_rank() == 0

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def synchronize_seed(local_rank, rank, seed=-1):
    if seed < 0:
        seed = get_true_random_32bit_positive_integer()

    if rank == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    seed_tensor = seed_tensor.cuda(local_rank)

    comm.broadcast(tensor=seed_tensor, src=0)

    seed = int(seed_tensor.item()) + rank

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={rank}")
    return seed

seed = synchronize_seed(local_rank, rank)
log_all(f"rank = {rank}, local_rank = {local_rank}, world_size = {world_size}, seed={seed}")

# Load the dataset
dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")

# Load the tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

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
