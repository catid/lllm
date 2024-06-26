# TODO:
# * Get FSDP training working
# * Periodically print validation loss / example output
# * 1.5M tokens/second training is the benchmark to beat
# * Use [Eluether harness to eval model](https://github.com/EleutherAI/lm-evaluation-harness)

import os, random, time, shutil, argparse, yaml, math, copy, sys
from packaging import version

import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from functools import partial

from model.model import LatentLanguage, LatentLanguageConfig, MultiQueryAttentionDConv

import numpy as np

import wandb

from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

import schedulefree
from adalomo import AdaLomo
from adam_mini import Adam_mini
import bitsandbytes as bnb

from logger_tt import setup_logging, logger

from torch.profiler import profile, record_function, ProfilerActivity

def get_current_script_directory():
    # Get the absolute path of the current script
    script_path = os.path.abspath(sys.argv[0])
    # Get the directory name from the script path
    script_dir = os.path.dirname(script_path)
    return script_dir

setup_logging(
    use_multiprocessing="fork",
    log_path=os.path.join(get_current_script_directory(), "train.log"))

# Enable cuDNN benchmarking to improve online performance
torch.backends.cudnn.benchmark = True

# Disable profiling to speed up training
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

is_main_process = False

def round_up_to_next_multiple_of_128(x):
    if x % 128 == 0:
        return x
    return x + (128 - (x % 128))

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F # Clear high bit
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def synchronize_seed(args):
    if args.seed < 0:
        args.seed = get_true_random_32bit_positive_integer()

    device = torch.device("cuda")

    if args.global_rank == 0:
        seed_tensor = torch.tensor(args.seed, dtype=torch.long, device=device)  # A tensor with the value to be sent
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long, device=device)  # A tensor to receive the value

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

def remove_oldest_checkpoint(args):
    checkpoint_files = [f for f in os.listdir(args.output_dir) if f.endswith('.pth')]
    checkpoint_files.sort()
    
    if len(checkpoint_files) > args.max_checkpoints:
        oldest_checkpoint = checkpoint_files[0]
        oldest_checkpoint_path = os.path.join(args.output_dir, oldest_checkpoint)
        os.remove(oldest_checkpoint_path)
        logger.info(f"Removed oldest checkpoint: {oldest_checkpoint_path}")

def save_checkpoint(args, model, optimizer, checkpoint_info):
    os.makedirs(args.output_dir, exist_ok=True)

    filename = f"checkpoint_epoch_{checkpoint_info['epoch']}_step_{checkpoint_info['step']}.pth"
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

    remove_oldest_checkpoint(args)

def load_checkpoint(args, model, optimizer):
    try:
        yml_path = os.path.join(args.output_dir, "latest.yml")
        with open(yml_path, 'r') as file:
            checkpoint_info = yaml.safe_load(file)
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
    device = torch.device("cuda")

    model.train()

    # Training statistics
    sum_loss = torch.tensor(0.0, device=device)
    sum_tokens = torch.tensor(0, device=device)
    sum_correct = torch.tensor(0, device=device)

    if args.optimizer == "adalomo" and args.grad_accum != 1:
        raise ValueError(f"AdaLOMO does not support gradient accumulation")

    # FIXME: Run parallel sums to reduce memory usage for grad_accum
    for grad_accum_step in range(args.grad_accum):
        batch, is_cont, step, total_steps = dataloader.get_micro_batch()

        if batch is None:
            return None, None

        input_ids = torch.from_numpy(batch).to(torch.long).to(device)
        labels = input_ids[..., :-2].contiguous()
        targets_1 = input_ids[..., 1:-1].contiguous()
        targets_2 = input_ids[..., 2:].contiguous()

        sum_tokens += torch.sum(targets_1 != -1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(labels, targets_1, targets_2)
            loss = loss / args.grad_accum
            sum_loss += loss.detach()

        predictions = torch.argmax(logits, dim=-1)
        sum_correct += torch.sum((predictions == targets_1) & (targets_1 != -1))

        if args.optimizer != "adalomo":
            if args.shard_strategy == "NO_SHARD":
                model.require_backward_grad_sync = (grad_accum_step + 1 >= args.grad_accum)
            loss.backward()

    lr = get_lr(args, step)

    if args.optimizer == "adalomo":
        optimizer.grad_norm(loss)
        optimizer.fused_backward(loss, lr)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Schedulefree optimizer does not support lr scheduling
        if args.optimizer != "schedulefree":
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Sync statistics between ranks
    dist.all_reduce(tensor=sum_loss, op=dist.ReduceOp.AVG)
    dist.all_reduce(tensor=sum_tokens, op=dist.ReduceOp.SUM)
    dist.all_reduce(tensor=sum_correct, op=dist.ReduceOp.SUM)

    # Note: sum_loss is now average loss
    return sum_loss.item(), sum_tokens.item(), sum_correct.item(), lr

# learning rate decay scheduler (linear warmup, constant LR, and 1-sqrt cooldown)
# Following results from "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" https://arxiv.org/abs/2405.18392v1
# Take-away: You can keep the learning rate constant until you decide to end training at any point after a short cooldown period.
# They recommend a cooldown period that is 20% of the training steps.
def get_lr(args, step):
    assert step <= args.steps
    # 1) linear warmup for args.warmup steps
    if step < args.warmup:
        return args.lr * (step+1) / args.warmup
    # 2) constant lr for a while
    elif step < args.steps - args.cooldown:
        return args.lr
    # 3) 1-sqrt cooldown for args.cooldown steps
    else:
        decay_ratio = (step - (args.steps - args.cooldown)) / args.cooldown
        return args.lr * (1 - math.sqrt(decay_ratio))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_training_state_info(model, optimizer):
    print("Model state information:")
    total_model_size = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  Layer: {name}")
            print(f"    Size: {param.size()}")
            print(f"    Data type: {param.dtype}")
            total_model_size += param.numel() * param.element_size()
    print(f"Total model size: {total_model_size} bytes")

    print("\nOptimizer state information:")
    total_optimizer_size = 0
    for group in optimizer.param_groups:
        print(f"  Parameter group:")
        for param_id, param in enumerate(group['params']):
            print(f"    Parameter {param_id}:")
            print(f"      Size: {param.size()}")
            print(f"      Data type: {param.dtype}")
            total_optimizer_size += param.numel() * param.element_size()
            state = optimizer.state[param]
            for key, value in state.items():
                if torch.is_tensor(value):
                    print(f"      State '{key}':")
                    print(f"        Size: {value.size()}")
                    print(f"        Data type: {value.dtype}")
                    total_optimizer_size += value.numel() * value.element_size()
    print(f"Total optimizer state size: {total_optimizer_size} bytes")

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f'Memory allocated: {allocated / 1024**2:.2f} MB')
    print(f'Memory reserved: {reserved / 1024**2:.2f} MB')

    print(torch.cuda.memory_summary())

def setup_fsdp(args, model):
    # Note: torch.compile does not seem to improve performance
    if args.compile:
        model = torch.compile(model, dynamic=True, fullgraph=False)

    if args.shard_strategy == "NO_SHARD":
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        return model

    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and version.parse(torch.version.cuda) >= version.parse("11.0")
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )

    if bf16_ready:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        logger.warn("BF16 not available")
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    t5_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                MultiQueryAttentionDConv,
            },
        )

    # Select the sharding strategy based on the argument
    if args.shard_strategy == "FULL_SHARD":
        shard_strategy = ShardingStrategy.FULL_SHARD
    elif args.shard_strategy == "SHARD_GRAD_OP":
        shard_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.shard_strategy == "HYBRID_SHARD":
        shard_strategy = ShardingStrategy.HYBRID_SHARD
    elif args.shard_strategy == "NO_SHARD":
        shard_strategy = ShardingStrategy.NO_SHARD
    else:
        raise ValueError(f"Unknown sharding strategy: {args.shard_strategy}")

    # Required for torch.compile compatibility (but uses a bit more memory)
    use_orig_params = args.compile

    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        use_orig_params=use_orig_params,
        sharding_strategy=shard_strategy)

    model = DistributedDataParallel(model)

    return model

def main(args, shard_config):
    logger.info(f"Node local_rank={args.local_rank} global_rank={args.global_rank} local_world_size={args.local_world_size} world_size={args.world_size}")

    dist.init_process_group(backend='nccl', rank=args.global_rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    if is_main_process:
        logger.info(f"Arguments: {args}")

    cfg = LatentLanguageConfig()
    cfg.n_vocab = round_up_to_next_multiple_of_128(shard_config["n_vocab"] + 1)
    cfg.padding_token = shard_config["n_vocab"] # Use largest token value as padding token

    cfg.block_size = args.context
    cfg.bnb_embedding = (args.optimizer == "adamw8bit")

    # Register model parameters with bitsandbytes before copying to GPU
    mng = bnb.optim.GlobalOptimManager.get_instance()
    model = LatentLanguage(cfg)

    mng.register_parameters(model.parameters())

    model = model.to(device).to(torch.bfloat16)

    if args.optimizer == "schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adalomo":
        optimizer = AdaLomo(
            model,
            lr=args.lr,
            clip_grad_norm=1.0,
            clip_grad_value=4.0,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adam_mini":
        optimizer = Adam_mini(
            model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            zero_3=False,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_query_groups=1) # 1=MQA
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            amsgrad=False,
            optim_bits=32,
            min_8bit_size=16384,
            percentile_clipping=100,
            block_wise=True,
            is_paged=False)
        model.override_config(mng)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    model = setup_fsdp(args, model)

    params = count_parameters(model)
    if is_main_process:
        logger.info(f"Total model parameters: {params/1000000.0:.2f}M")

    if args.verify_dataset:
        if is_main_process:
            logger.info("Verifying dataset... (should take about one minute per 0.4T tokens using an SSD)")

            is_valid = DataVerifier.verify(args.dataset_dir)

            if not is_valid:
                raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    if is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    dist.barrier()

    args.seed = synchronize_seed(args)

    # Weights & Biases
    if args.wandb and is_main_process:
        if not args.name:
            raise RuntimeError("The --name argument is required when using --wandb")
        wandb.init(project=args.project, name=args.name, config=args)
        wandb.run.log_code = False

    step = 0
    epoch = 0
    tokens = 0
    total_steps = 0
    wallclock_time = 0.0

    if args.resume:
        checkpoint_info = load_checkpoint(args, model, optimizer)
        if checkpoint_info is not None:
            step = checkpoint_info.get("step", 0)
            epoch = checkpoint_info.get("epoch", 0)
            total_steps = checkpoint_info.get("total_steps", 0)
            tokens = checkpoint_info.get("tokens", 0)
            wallclock_time = checkpoint_info.get("wallclock_time", 0.0)
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

    while total_steps < args.steps:
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
        validation_config.start_step = 0
        validation_dataloader.begin_epoch(validation_config)

        while True:
            start_time = time.time()

            avg_train_loss, sum_tokens, sum_correct, lr = train_one_step(
                            args, optimizer, model, dataloader)

            if avg_train_loss is None:
                logger.info(f"Epoch {epoch} data exhausted on global_rank={args.global_rank} at step={step}")
                break

            end_time = time.time()
            step_time = end_time - start_time

            tokens += sum_tokens
            tokens_per_second = sum_tokens / step_time
            correct_pct = sum_correct * 100.0 / sum_tokens

            if is_main_process:
                logger.info(f"Step {step}: AvgLoss={avg_train_loss:.4f} StepTime={step_time:.2f} sec correct={correct_pct:.2f}% Tokens={tokens/1000000.0:.2f}M at {tokens_per_second/1000.0:.2f} ktokens/sec") 

            step += 1
            total_steps += 1
            wallclock_time += step_time

            if step % args.checkpoint_interval == 0 or step >= args.steps:
                checkpoint_info = {
                    'train_version': 1,
                    'avg_train_loss': avg_train_loss,
                    'args': vars(args),
                    'total_steps': total_steps,
                    'epoch': epoch,
                    'step': step,
                    'tokens': tokens,
                    'wallclock_time': wallclock_time,
                    'lr': lr,
                }

                if args.wandb and is_main_process:
                    wandb.log(checkpoint_info)

                save_checkpoint(args, model, optimizer, checkpoint_info)

                # Avoid fragmentation-related OOM by releasing cache
                torch.cuda.empty_cache()

            #if is_main_process:
                #print_training_state_info(model, optimizer)

            if step >= args.steps:
                logger.info(f"Training complete.  Total steps: {total_steps}")
                dist.barrier()
                return

        # End of epoch
        logger.info(f"Epoch {epoch} complete on rank {args.global_rank}")
        epoch += 1
        step = 0
        dist.barrier()

    logger.info(f"Training complete.  Total steps: {total_steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Config
    parser.add_argument("--output-dir", type=str, default="~/lllm_output", help="Path to model checkpoints for resuming") 
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random numbers.  Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")

    # Dataset
    parser.add_argument("--dataset-dir", type=str, default="~/dataset_shard", help="Dataset directory")
    parser.add_argument("--holdout-dir", type=str, default="~/holdout_shard", help="Holdout directory")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify the dataset before training")
    parser.add_argument("--min-data-length", type=int, default=64, help="Minimum data length to use for training")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--project", type=str, default="my_project", help="Collection of experiments on wandb")

    # Hyperparameters
    parser.add_argument("--context", type=int, default=2050, help="Context size for each microbatch")
    parser.add_argument("--microbatch", type=int, default=6, help="Microbatch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--optimizer", type=str, default="adamw8bit", help="Options: schedulefree, adalomo, adam_mini, adamw8bit, adamw")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.3, help="Weight decay for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")
    parser.add_argument("--steps", type=int, default=160000, help="Total steps for training")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps (recommended 2000 above 100k steps)")
    parser.add_argument("--cooldown", type=int, default=32000, help="Cooldown steps (recommended 0.2x total steps)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient magnitude")

    # Checkpointing
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Steps between checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=20, help="Number of checkpoitns to keep on disk")

    # Distributed training
    parser.add_argument("--shard-strategy", type=str, default="NO_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"], help="Sharding strategy for FSDP")

    args = parser.parse_args()

    # Get environment variables from torchrun
    args.global_rank = int(os.getenv("RANK", "0"))
    is_main_process = args.global_rank == 0
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    args.dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    args.holdout_dir = os.path.abspath(os.path.expanduser(args.holdout_dir))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

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

    dist.destroy_process_group()
