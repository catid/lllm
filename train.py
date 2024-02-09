################################################################################
# Torch

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os, random

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

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

def synchronize_seed(args, rank, shard_id):
    if args.seed < 0:
        seed = get_true_random_32bit_positive_integer()
    else:
        seed = args.seed

    if shard_id == 0:
        seed_tensor = torch.tensor(seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    seed_tensor = seed_tensor.cuda(rank)

    comm.broadcast(tensor=seed_tensor, src=0)

    seed = int(seed_tensor.item()) + shard_id
    args.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={shard_id}")
    return seed


################################################################################
# Model Training

from model.model import LatentLanguage, LatentLanguageConfig
from dataloader import DatasetLoader

def make_optimizer(model, args):
    model_parameters = list(model.parameters())
    optimizer_params = [p for p in model_parameters if not hasattr(p, "_optim")]
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def print_model_size(model, model_name):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_0(f"{model_name} has {num_parameters:,} trainable parameters")

def ref_forward_and_loss(token_batch, mask_batch, model, criterion):
    # DeepSpeed: forward + backward + optimize
    outputs = model(token_batch, mask_batch)
    return criterion(pred_image, target_image), pred_image

def train_model(args):
    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    cfg = LatentLanguageConfig()

    model = LatentLanguage(cfg)

    model_optimizer = make_optimizer(model, args)
    model_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=args.max_epochs)

    # DeepSpeed engine
    lm_engine, lm_engine_optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=model_optimizer,
        lr_scheduler=model_lr_scheduler,
        #config_params=args.deepspeed_config,  <- This should be in the args
        model_parameters=model.parameters())

    log_0(f"Arguments: {args}")

    comm.barrier()

    fp16 = lm_engine.fp16_enabled()
    log_0(f'decoder_engine.fp16_enabled={fp16}')

    if fp16:
        image_dtype = torch.float16
    else:
        image_dtype = torch.float32

    rank = lm_engine.local_rank
    shard_id = lm_engine.global_rank
    num_gpus = lm_engine.world_size
    train_batch_size = lm_engine.train_batch_size()
    data_loader_batch_size = lm_engine.train_micro_batch_size_per_gpu()
    steps_per_print = lm_engine.steps_per_print()

    num_loader_threads = os.cpu_count()//2
    crop_w = 224
    crop_h = 224
    val_split_ratio = 0.2
    min_train_size = 1024
    min_val_size = 128

    seed = synchronize_seed(args, rank, shard_id)

    log_all(f"rank = {rank}, num_shards = {num_gpus}, shard_id={shard_id}, train_batch_size = {train_batch_size}, data_loader_batch_size = {data_loader_batch_size}, steps_per_print = {steps_per_print}, seed={seed}")

    # Define L1 Loss and Optimizer
    criterion = torch.nn.L1Loss()
    criterion.cuda(rank)

    forward_and_loss = torch.compile(ref_forward_and_loss, dynamic=True, fullgraph=True)

    dataset_loader = DatasetLoader("wikitext", "wikitext-103-raw-v1")
    train_dataloader = dataset_loader.get_dataloader("train")
    test_dataloader = dataset_loader.get_dataloader("test")

    # Your training loop here
    num_epochs = 10
    device = "cuda"

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            loss = forward_and_loss(input_ids, attention_mask)

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # Your evaluation code here


################################################################################
# Entrypoint

import argparse

def main(args):
    train_model(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=1e6)
    parser.add_argument('--interval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--max-epochs', type=int, default=128)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    main(args)
