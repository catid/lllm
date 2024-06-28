# TODO:
# * Get FSDP training working
# * Periodically print validation loss / example output
# * 1.5M tokens/second training is the benchmark to beat
# * Use [Eluether harness to eval model](https://github.com/EleutherAI/lm-evaluation-harness)

import os, random, time, shutil, argparse, yaml, math, copy, sys
from packaging import version

import torch
from torch import nn

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

from model.model import LatentLanguage, LatentLanguageConfig, MultiQueryAttentionDConv

import numpy as np

import wandb
import json

from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

from torch.profiler import profile, record_function, ProfilerActivity

import schedulefree

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def get_current_script_directory():
    # Get the absolute path of the current script
    script_path = os.path.abspath(sys.argv[0])
    # Get the directory name from the script path
    script_dir = os.path.dirname(script_path)
    return script_dir

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
        # A tensor with the value to be sent
        seed_tensor = torch.tensor(args.seed, dtype=torch.long, device=device)
    else:
        # A tensor to receive the value
        seed_tensor = torch.zeros(1, dtype=torch.long, device=device)

    comm.broadcast(tensor=seed_tensor, src=0)

    # Seed PRNGs using the shared seed
    seed = int(seed_tensor.item())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_all(f"Using seed: {seed} for shard_id={args.global_rank}")
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
        log_all(f"Removed oldest checkpoint: {oldest_checkpoint_path}")

def save_deepspeed_model_engine(model_engine, args, checkpoint_info):
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"checkpoint_epoch_{checkpoint_info['epoch']}_step_{checkpoint_info['start_step']}.pth"
    checkpoint_path = os.path.join(args.output_dir, filename)

    # Remove module. prefix from keys
    saved_state_dict = model_engine.state_dict()
    fixed_state_dict = {key.replace("module.", ""): value for key, value in saved_state_dict.items()}

    # Add our data to the state dict to facilitate the evaluation script
    fixed_state_dict['lllm'] = checkpoint_info

    torch.save(fixed_state_dict, args.output_model)

    checkpoint_info["checkpoint_path"] = checkpoint_path

    yml_path = os.path.join(args.output_dir, "latest.yml")
    with open(yml_path, 'w') as file:
        yaml.dump(checkpoint_info, file, default_flow_style=False)

    log_all(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(args, model_engine, optimizer):
    try:
        yml_path = os.path.join(args.output_dir, "latest.yml")
        with open(yml_path, 'r') as file:
            checkpoint_info = yaml.safe_load(file)
        checkpoint_path = checkpoint_info["checkpoint_path"]

        checkpoint = torch.load(checkpoint_path)
        model_engine.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_info = checkpoint['info']
    except Exception as e:
        log_all(f"Error loading checkpoint: {e}")
        return None
    return checkpoint_info

def train_one_step(args, optimizer, model_engine, dataloader, step):
    device = torch.device("cuda")

    model_engine.train()

    # Training statistics
    sum_loss = torch.tensor(0.0, device=device)
    sum_tokens = torch.tensor(0, device=device)
    sum_correct = torch.tensor(0, device=device)

    for grad_accum_step in range(args.grad_accum):
        batch, is_cont, data_step, total_steps = dataloader.get_micro_batch()

        if batch is None:
            return None, None

        input_ids = torch.from_numpy(batch).to(torch.long).to(device)
        labels = input_ids[..., :-2].contiguous()
        targets_1 = input_ids[..., 1:-1].contiguous()
        targets_2 = input_ids[..., 2:].contiguous()

        sum_tokens += torch.sum(targets_1 != -1)

        logits, loss = model_engine(labels, targets_1, targets_2)
        sum_loss += loss.detach()

        predictions = torch.argmax(logits, dim=-1)
        sum_correct += torch.sum((predictions == targets_1) & (targets_1 != -1))

        model_engine.backward(loss)

        model_engine.step()

    # Sync statistics between ranks
    comm.all_reduce(tensor=sum_loss, op=comm.ReduceOp.AVG)
    comm.all_reduce(tensor=sum_tokens, op=comm.ReduceOp.SUM)
    comm.all_reduce(tensor=sum_correct, op=comm.ReduceOp.SUM)

    # Note: sum_loss is now average loss
    return sum_loss.item(), sum_tokens.item(), sum_correct.item(), data_step

# learning rate decay scheduler (linear warmup, constant LR, and 1-sqrt cooldown)
# Following results from "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" https://arxiv.org/abs/2405.18392v1
# Take-away: You can keep the learning rate constant until you decide to end training at any point after a short cooldown period.
# They recommend a cooldown period that is 20% of the training steps.
def calculate_lr(args, step):
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

# DeepSpeed scheduler wrapping our LR scheduler
class CustomLR(object):
    def __init__(self, optimizer, args):
        self.args = args
        self.optimizer = optimizer
        self.last_batch_iteration = -1

    def get_lr(self):
        if self.last_batch_iteration < 0:
            log_all("ERROR: Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        lr = calculate_lr(self.args, self.last_batch_iteration)
        return [lr] * len(self.optimizer.param_groups)

    def get_last_lr(self):
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    def step(self, last_batch_iteration=None):
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration
        lr = calculate_lr(self.args, last_batch_iteration)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._last_lr = [lr] * len(self.optimizer.param_groups)

    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args, shard_config):
    deepspeed.init_distributed(dist_backend='nccl', verbose="false")

    cfg = LatentLanguageConfig()
    cfg.n_vocab = round_up_to_next_multiple_of_128(shard_config["n_vocab"] + 1)
    cfg.padding_token = shard_config["n_vocab"] # Use largest token value as padding token
    cfg.block_size = args.context
    cfg.bnb_embedding = False
    model = LatentLanguage(cfg)

    params = count_parameters(model)
    if is_main_process():
        log_0(f"Total model parameters: {params/1000000.0:.2f}M")

    if args.optimizer == "schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.optimizer == "schedulefree":
        scheduler = None
    else:
        scheduler = CustomLR(optimizer, args)

    # Modify deepspeed configuration programmatically
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = args.microbatch
    ds_config["gradient_accumulation_steps"] = args.grad_accum

    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and version.parse(torch.version.cuda) >= version.parse("11.0")
        and torch.cuda.nccl.version() >= (2, 10)
    )
    ds_config["fp16"]["enabled"] = not bf16_ready
    ds_config["bf16"]["enabled"] = bf16_ready

    # Remove deepspeed_config from the args (we pass a dict into deepspeed.initialize)
    args.deepspeed_config = None

    # DeepSpeed engine
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
        model_parameters=model.parameters())

    log_0(f"Arguments: {args} bf16_ready={bf16_ready}")

    if args.local_rank != model.local_rank:
        raise RuntimeError("Local rank does not match model local rank")
    args.local_world_size = shard_config["rank_count"]
    args.global_rank = model.global_rank
    args.world_size = model.world_size
    if model.train_micro_batch_size_per_gpu() != args.microbatch:
        raise RuntimeError(f"model.train_micro_batch_size_per_gpu()=={model.train_micro_batch_size_per_gpu()} does not match args.microbatch=={args.microbatch}")
    if model.train_batch_size() != args.microbatch * args.grad_accum * args.local_world_size:
        raise RuntimeError(f"model.train_batch_size()=={model.train_batch_size()} does not match args.microbatch * args.grad_accum=={args.microbatch * args.grad_accum * args.local_world_size}")
    #steps_per_print = model.steps_per_print()

    log_all(f"Node local_rank={args.local_rank} global_rank={args.global_rank} local_world_size={args.local_world_size} world_size={args.world_size}")

    if args.verify_dataset:
        if args.local_rank == 0:
            log_all("Verifying dataset... (should take about one minute per 0.4T tokens using an SSD)")

            is_valid = DataVerifier.verify(args.dataset_dir)

            if not is_valid:
                raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    args.seed = synchronize_seed(args)

    # Weights & Biases
    if args.wandb and is_main_process():
        if not args.name:
            raise RuntimeError("The --name argument is required when using --wandb")
        wandb.init(project=args.project, name=args.name, config=args)
        wandb.run.log_code = False

    model = model.to(device).to(torch.bfloat16)

    # Note: torch.compile does not seem to improve performance
    if args.compile:
        model = torch.compile(model, dynamic=True, fullgraph=False)

    if args.local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    comm.barrier()

    log_all("Starting training...")

    start_step = 0
    epoch = 0
    tokens = 0
    total_steps = 0
    wallclock_time = 0.0

    if args.resume:
        _, checkpoint_info = model.load_checkpoint(load_dir=args.output_dir)
        if checkpoint_info is not None:
            start_step = checkpoint_info.get("start_step", 0)
            epoch = checkpoint_info.get("epoch", 0)
            total_steps = checkpoint_info.get("total_steps", 0)
            tokens = checkpoint_info.get("tokens", 0)
            wallclock_time = checkpoint_info.get("wallclock_time", 0.0)
            log_all(f"Loaded checkpoint at start_step={start_step} epoch={epoch}")
        else:
            log_all("No checkpoint found - Starting training from scratch")
    else:
        if args.local_rank == 0:
            log_all("Resetting training - deleting output directory")
            recreate_folder(args.output_dir)
        comm.barrier()

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
        config.start_step = start_step
        dataloader.begin_epoch(config)

        validation_config = copy.deepcopy(config)
        validation_config.start_step = 0
        validation_dataloader.begin_epoch(validation_config)

        while True:
            start_time = time.time()

            avg_train_loss, sum_tokens, sum_correct, data_step = train_one_step(
                args,
                optimizer,
                model,
                dataloader,
                total_steps)

            if avg_train_loss is None:
                log_all(f"Epoch {epoch} data exhausted on global_rank={args.global_rank} at step={data_step}")
                break

            end_time = time.time()
            step_time = end_time - start_time

            tokens += sum_tokens
            tokens_per_second = sum_tokens / step_time
            correct_pct = sum_correct * 100.0 / sum_tokens

            if is_main_process():
                log_0(f"Step {total_steps}: AvgLoss={avg_train_loss:.4f} StepTime={step_time:.2f} sec correct={correct_pct:.2f}% Tokens={tokens/1000000.0:.2f}M at {tokens_per_second/1000.0:.2f} ktokens/sec") 

            start_step = data_step + 1
            total_steps += 1
            wallclock_time += step_time

            if total_steps % args.checkpoint_interval == 0 or total_steps >= args.steps:
                checkpoint_info = {
                    'train_version': 1,
                    'avg_train_loss': avg_train_loss,
                    'args': vars(args),
                    'total_steps': total_steps,
                    'epoch': epoch,
                    'start_step': start_step,
                    'tokens': tokens,
                    'wallclock_time': wallclock_time,
                    'lr': scheduler.get_last_lr()[0],
                }

                if args.wandb and is_main_process():
                    wandb.log(checkpoint_info)

                model.save_checkpoint(save_dir=args.output_dir, client_state=checkpoint_info)

                # Avoid fragmentation-related OOM by releasing cache
                torch.cuda.empty_cache()

            if total_steps >= args.steps:
                log_all(f"Training complete.  Total steps: {total_steps}")
                comm.barrier()
                return

        # End of epoch
        log_all(f"Epoch {epoch} complete on rank {args.global_rank}")
        epoch += 1
        start_step = 0
        comm.barrier()

    log_all(f"Training complete.  Total steps: {total_steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Config
    parser.add_argument("--output-dir", type=str, default="~/lllm_output", help="Path to model checkpoints for resuming") 
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random numbers.  Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", type=bool, default=False, help="Enable torch.compile")

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
    parser.add_argument("--microbatch", type=int, default=10, help="Microbatch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Options: adamw, schedulefree")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate for training")
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
    parser.add_argument("--local_rank", type=int, default=-1)

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

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
    if local_ranks != shard_config["rank_count"]:
        raise RuntimeError(f"Number of GPUs ({local_ranks}) does not match rank_count from shard config ({shard_config['rank_count']})")

    log_all(f"Shard config: {shard_config}")

    main(args, shard_config)
