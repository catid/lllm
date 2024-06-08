# Set the environment variables
import os

from model.model import LatentLanguage, LatentLanguageConfig

import torch
import torch.nn as nn

import numpy as np
import random, time, json
import shutil
import argparse
import yaml

import wandb

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

from cpp_dataloader import DataLoader, DataVerifier
#from mora import MoRALayer, merge_mora_weights, replace_linear_with_mora

#from sophia import SophiaG
import schedulefree

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

# Deepspeed logging functions
def log_0(msg):
    log_dist(msg, ranks=[0])
def log_all(msg):
    log_dist(msg, ranks=[-1])

def is_main_process():
    return comm.get_rank() == 0

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

    seed = int(seed_tensor.item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={rank}")
    return seed

def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def train_one_step(optimizer, criterion, model_engine, dataloader):
    model_engine.train()

    batch, is_cont = dataloader.get_micro_batch()

    if batch is None:
        return None

    input_ids = torch.from_numpy(batch).to(torch.long).to(model_engine.device)
    labels = input_ids[..., :-1].contiguous()
    targets = input_ids[..., 1:].contiguous()

    _, loss = model_engine(labels, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def save_deepspeed_model_engine(model_engine, fp16, args):
    # Write output .pth file
    saved_state_dict = model_engine.state_dict()

    # Remove module. prefix from keys
    fixed_state_dict = {key.replace("module.", ""): value for key, value in saved_state_dict.items()}

    # Add our data to the state dict to facilitate the evaluation script
    fixed_state_dict['lllm'] = {
        'fp16': fp16,
    }

    torch.save(fixed_state_dict, args.output_model)

def main(args, shard_config):
    t0 = time.time()

    # Initialize DeepSpeed
    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    cfg = LatentLanguageConfig()
    cfg.n_vocab = shard_config["n_vocab"]
    cfg.block_size = args.context
    model = LatentLanguage(cfg)

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    # Modify deepspeed configuration programmatically
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    ds_config["fp16"]["enabled"] = not args.fp32

    # Remove deepspeed_config from the args (we pass a dict into deepspeed.initialize)
    args.deepspeed_config = None

    # DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=None, # We do not use LR schedulers
        config=ds_config,
        model_parameters=model.parameters())

    log_0(f"Arguments: {args}")

    if args.verify_dataset:
        log_all("Verifying dataset... (should take about one minute per 0.4T tokens using an SSD)")

        if is_main_process():
            is_valid = DataVerifier.verify(args.dataset_dir)

            if not is_valid:
                raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    if is_main_process():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    comm.barrier()

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    log_all(f"Node rank={rank}, num_shards={num_gpus}, shard_id={shard_id}, train_batch_size={train_batch_size}, data_loader_batch_size={data_loader_batch_size}, steps_per_print={steps_per_print}")  

    seed = synchronize_seed(rank, shard_id)

    # Weights & Biases
    if args.wandb and is_main_process():
        if not args.name:
            raise "The --name argument is required when using --wandb"
        wandb.init(project=args.project, name=args.name, config=args)
        wandb.run.log_code = False

    criterion = nn.CrossEntropyLoss()
    criterion.cuda(rank)

    if args.compile:
        forward_and_loss = torch.compile(forward_and_loss, dynamic=True, fullgraph=False)

    start_epoch = 0

    if args.resume:
        _, client_state = model_engine.load_checkpoint(load_dir=args.output_dir)
        if client_state is not None:
            start_epoch = client_state['epoch'] + 1
            log_all(f"Loaded checkpoint at epoch {client_state['epoch']}")
        else:
            log_all("No checkpoint found - Starting training from scratch")
    else:
        log_0("Resetting training - deleting output directory")
        if rank == 0:
            delete_folder_contents(args.output_dir)
        comm.barrier()

    # DataLoader
    dataloader = DataLoader(args.dataset_dir, rank=rank, local_ranks=shard_config["rank_count"])

    for epoch in range(start_epoch, args.max_epochs):
        # This seed is synchronized between ranks so they do not reuse the same data
        seed2 = epoch
        dataloader.start_epoch(seed, seed2, data_loader_batch_size, args.context)

        while True:

            start_time = time.time()

            train_loss = train_one_step(optimizer, criterion, model_engine, dataloader)

            if train_loss is None:
                break

            end_time = time.time()
            epoch_time = end_time - start_time

            # Sync variables between ranks
            avg_train_loss = torch.tensor(train_loss).cuda(rank)
            comm.all_reduce(tensor=avg_train_loss, op=comm.ReduceOp.AVG)

            if is_main_process():
                log_0(f"Step complete - TrainLoss={avg_train_loss:.4f} Time={epoch_time:.2f} sec")

        if is_main_process():
            log_0(f"Epoch {epoch + 1} - TrainLoss={avg_train_loss:.4f} Time={epoch_time:.2f} sec")

            if args.wandb:
                lr = optimizer.param_groups[0]['lr']
                wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch, "wallclock_time": epoch_time, "lr": lr})

    if is_main_process():
        t1 = time.time()
        dt = t1 - t0

        log_0(f'Training complete after {dt} seconds.  Best model was written to {args.output_model}')

        if args.wandb:
            wandb.finish()

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Deepspeed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Path to the latest checkpoint for resuming")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--output-model", type=str, default="cifar10.pth", help="Output model file name")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="~/dataset_shard", help="Dataset directory")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify the dataset before training")

    # Misc
    parser.add_argument("--seed", type=int, default=-1, help="Seed for random numbers.  Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--fp32", action='store_true', help="Enable fp32 training (fp16 default)")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--project", type=str, default="my_project", help="Collection of experiments on wandb")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.3, help="Weight decay for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")
    parser.add_argument("--context", type=int, default=1024, help="Context size for each microbatch")

    # Training duration
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=50, help="Patience for validation loss not decreasing before early stopping")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

    args.dataset_dir = os.path.expanduser(args.dataset_dir)

    if not os.path.exists(args.dataset_dir):
        raise RuntimeError(f"Dataset directory {args.dataset_dir} does not exist")

    args_path = os.path.join(args.dataset_dir, "args.yml")
    shard_config = read_yaml_file(args_path)

    local_ranks = torch.cuda.device_count()
    if local_ranks != shard_config["rank_count"]:
        raise RuntimeError(f"Number of GPUs ({local_ranks}) does not rank_count from shard config ({shard_config['rank_count']})")

    log_all(f"Shard config: {shard_config}")

    main(args, shard_config)
