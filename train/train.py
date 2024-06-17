import os

from model.model import LatentLanguage, LatentLanguageConfig

import torch

import numpy as np
import random, time
import shutil
import argparse
import yaml
import math
import copy

import wandb

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig, OptimStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.api import init_process_group, destroy_process_group

import torch.distributed as dist

from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

import schedulefree

from logger_tt import setup_logging, logger

setup_logging(use_multiprocessing="fork")

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

def is_main_process():
    return dist.get_rank() == 0

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def synchronize_seed(args):
    if args.seed < 0:
        args.seed = get_true_random_32bit_positive_integer()

    if args.global_rank == 0:
        seed_tensor = torch.tensor(args.seed, dtype=torch.long)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long)  # A tensor to receive the value

    dist.broadcast(tensor=seed_tensor, src=0)

    # Seed PRNGs using the shared seed
    seed = int(seed_tensor.item())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"Using seed: {seed} for shard_id={args.global_rank}")
    return seed

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def save_checkpoint(args, model, optimizer, checkpoint_info):
    os.makedirs(args.output_dir, exist_ok=True)

    filename = f"checkpoint_epoch_{checkpoint_info["epoch"]}_step_{checkpoint_info["resume_step"]}.pth"
    checkpoint_path = os.path.join(args.output_dir, filename)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'info': checkpoint_info
    }
    torch.save(checkpoint, checkpoint_path)

    checkpoint_info["checkpoint_path"] = checkpoint_path

    yml_path = os.path.join(args.output_dir, "latest.yml")
    with open(yml_path, 'w') as file:
        yaml.dump(checkpoint_info, file, default_flow_style=False)

    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(args, model, optimizer):
    try:
        yml_path = os.path.join(args.output_dir, "latest.yml")
        with open(args_path, 'r') as file:
            checkpoint_info = yaml.safe_load(yml_path)
        checkpoint_path = checkpoint_info["checkpoint_path"]

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_info = checkpoint['info']
    except Exception as e:
        logger.info(f"Error loading checkpoint: {e}")
        return None
    return checkpoint_info

def train_one_step(args, optimizer, model, dataloader):
    model.train()
    loss_accum = 0.0

    for grad_accum_step in range(args.grad_accum):
        batch, is_cont, step, total_steps = dataloader.get_micro_batch()

        if batch is None:
            return None, None

        input_ids = torch.from_numpy(batch).to(torch.long).to(model.device)
        labels = input_ids[..., :-1].contiguous()
        targets = input_ids[..., 1:].contiguous()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(labels, targets)
            loss = loss / args.grad_accum
            loss_accum += loss.detach()

        # TBD: Only for DDP
        model.require_backward_grad_sync = (grad_accum_step + 1 == args.grad_accum)
        loss.backward()

        tokens_trained = torch.sum(targets != -1).item()

        # FIXME
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = (predictions == targets) & (targets != -1)
        correct_tokens_count = torch.sum(correct_predictions).item()

    # TBD: Only for DDP
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    lr = get_lr(args, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return loss.item(), tokens_trained, correct_tokens_count

# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(args, step):
    assert step <= args.steps
    # 1) linear warmup for warmup_iters steps
    if step < args.warmup:
        return args.lr * (step+1) / args.warmup
    # 2) constant lr for a while
    elif step < args.steps - args.cooldown:
        return args.lr
    # 3) 1-sqrt cooldown
    else:
        decay_ratio = (step - (args.steps - args.cooldown)) / args.cooldown
        return args.lr * (1 - math.sqrt(decay_ratio))

def main(args, shard_config):
    if is_main_process():
        logger.info(f"Arguments: {args}")

    logger.info(f"Node local_rank={args.local_rank} global_rank={args.global_rank} local_world_size={args.local_world_size} world_size={args.world_size}")

    dist.init_process_group(backend='nccl', rank=args.global_rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    cfg = LatentLanguageConfig()
    cfg.n_vocab = shard_config["n_vocab"]
    cfg.block_size = args.context
    model = LatentLanguage(cfg)

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    if args.compile:
        model = torch.compile(model, dynamic=True, fullgraph=False)

    model = FSDP(model)

    if args.verify_dataset:
        logger.info("Verifying dataset... (should take about one minute per 0.4T tokens using an SSD)")

        if is_main_process():
            is_valid = DataVerifier.verify(args.dataset_dir)

            if not is_valid:
                raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    if is_main_process():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    dist.barrier()

    args.seed = synchronize_seed(args)

    # Weights & Biases
    if args.wandb and is_main_process():
        if not args.name:
            raise "The --name argument is required when using --wandb"
        wandb.init(project=args.project, name=args.name, config=args)
        wandb.run.log_code = False

    step = 0
    epoch = 0
    tokens = 0

    if args.resume:
        checkpoint_info = load_checkpoint(args, model, optimizer)
        if checkpoint_info is not None:
            step = checkpoint_info["resume_step"]
            epoch = checkpoint_info["epoch"]
            tokens = checkpoint_info["tokens"]
            logger.info(f"Loaded checkpoint at step={step} epoch={epoch}")
        else:
            logger.info("No checkpoint found - Starting training from scratch")
    else:
        if args.local_rank == 0:
            logger.info("Resetting training - deleting output directory")
            recreate_folder(args.output_dir)
        dist.barrier()

    # DataLoader
    dataloader = DataLoader(args.dataset_dir)
    if not dataloader:
        raise RuntimeError("Dataloader failed to initialize")
    validation_dataloader = DataLoader(args.holdout_dir)
    if not validation_dataloader:
        raise RuntimeError("Validation dataloader failed to initialize")

    while True:
        config = EpochConfig()
        config.seed0 = args.seed
        config.seed1 = epoch
        config.local_rank = args.local_rank
        config.local_rank_count = shard_config["rank_count"]
        config.padding_token = -1
        config.micro_batch_size = args.microbatch
        config.context_size = args.context
        config.min_data_length = args.min_data_length
        config.start_step = step
        dataloader.begin_epoch(config)

        validation_config = copy.deepcopy(config)
        validation_config.start_step = 0 # 
        validation_dataloader.begin_epoch(validation_config)

        while True:
            start_time = time.time()

            train_loss, train_tokens, correct_tokens_count = train_one_step(
                args, optimizer, model, dataloader)

            if train_loss is None:
                logger.info(f"Epoch {epoch} data exhausted on global_rank={args.global_rank} at step={step}")
                break

            end_time = time.time()
            step_time = end_time - start_time

            # Sync variables between ranks
            avg_train_loss = torch.tensor(train_loss)
            sum_tokens = torch.tensor(train_tokens)
            sum_correct = torch.tensor(correct_tokens_count)
            dist.all_reduce(tensor=avg_train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(tensor=sum_tokens, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor=sum_correct, op=dist.ReduceOp.SUM)
            avg_train_loss = avg_train_loss.item()
            sum_tokens = sum_tokens.item()
            sum_correct = sum_correct.item()

            tokens += sum_tokens
            tokens_per_second = sum_tokens / step_time
            correct_pct = sum_correct * 100.0 / sum_tokens

            if is_main_process():
                logger.info(f"Step {step}: AvgLoss={avg_train_loss:.4f} StepTime={step_time:.2f} sec correct={correct_pct:.2f}% Tokens={tokens/1000000.0:.2f}M at {tokens_per_second/1000.0:.2f} ktokens/sec") 

            step += 1

            if step % args.checkpoint_interval == 0:
                checkpoint_info = {
                    'train_version': 1,
                    'avg_train_loss': avg_train_loss,
                    'args': args,
                    'epoch': epoch,
                    'step': step,
                    'tokens': tokens,
                    'wallclock_time': step_time,
                    'lr': optimizer.param_groups[0]['lr'],
                }

                if args.wandb:
                    wandb.log(checkpoint_info)

                save_checkpoint(args, model, optimizer, checkpoint_info)

        logger.info(f"Epoch {epoch} complete on rank {args.global_rank}")

        dist.barrier()

        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Config
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Path to model checkpoints for resuming") 
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random numbers.  Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="~/dataset_shard", help="Dataset directory")
    parser.add_argument("--holdout-dir", type=str, default="~/holdout_shard", help="Holdout directory")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify the dataset before training")
    parser.add_argument("--context", type=int, default=4096, help="Context size for each microbatch")
    parser.add_argument("--min-data-length", type=int, default=64, help="Minimum data length to use for training")
    parser.add_argument("--microbatch", type=int, default=8, help="Microbatch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--project", type=str, default="my_project", help="Collection of experiments on wandb")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.3, help="Weight decay for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")
    parser.add_argument("--steps", type=int, default=1000000, help="Total steps for training")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--cooldown", type=int, default=100000, help="Cooldown steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient magnitude")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Steps between checkpoints")

    args = parser.parse_args()

    # Get environment variables from torchrun
    args.global_rank = int(os.getenv("RANK", "0"))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    args.local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    args.dataset_dir = os.path.expanduser(args.dataset_dir)

    if not os.path.exists(args.dataset_dir):
        raise RuntimeError(f"Dataset directory {args.dataset_dir} does not exist")
    if not os.path.exists(args.holdout_dir):
        raise RuntimeError(f"Holdout directory {args.holdout_dir} does not exist")

    args_path = os.path.join(args.dataset_dir, "args.yml")
    with open(args_path, 'r') as file:
        shard_config = yaml.safe_load(file)

    local_ranks = torch.cuda.device_count()
    if local_ranks != args.local_world_size:
        raise RuntimeError(f"Number of GPUs ({local_ranks}) does not match torchrun local_world_size ({args.local_world_size})")
    if local_ranks != shard_config["rank_count"]:
        raise RuntimeError(f"Number of GPUs ({local_ranks}) does not match rank_count from shard config ({shard_config['rank_count']})")

    logger.info(f"Shard config: {shard_config}")

    main(args, shard_config)

    destroy_process_group()

