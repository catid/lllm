import os, random, time, shutil, argparse, yaml, math, copy, sys
from packaging import version

import torch
import torch.distributed as dist
import deepspeed
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from model.model import LatentLanguage, LatentLanguageConfig, MultiQueryAttentionDConv

import numpy as np

import wandb

from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

import schedulefree
from lomo_optim import AdaLomo

from logger_tt import setup_logging, logger

def get_current_script_directory():
    script_path = os.path.abspath(sys.argv[0])
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
        seed_tensor = torch.tensor(args.seed, dtype=torch.long, device=device)
    else:
        seed_tensor = torch.zeros(1, dtype=torch.long, device=device)

    dist.broadcast(tensor=seed_tensor, src=0)

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

def setup_deepspeed(args, model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    ds_config = {
        "train_batch_size": args.microbatch * args.grad_accum * args.world_size,
        "train_micro_batch_size_per_gpu": args.microbatch,
        "steps_per_print": args.checkpoint_interval,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup,
            }
        },
        "gradient_clipping": args.grad_clip,
        "fp16": {
            "enabled": True,
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=parameters,
        config=ds_config
    )

    return model_engine, optimizer

def train_one_step(args, model_engine, dataloader):
    model_engine.train()

    sum_loss = 0.0
    sum_tokens = 0
    sum_correct = 0

    for _ in range(args.grad_accum):
        batch, is_cont, step, total_steps = dataloader.get_micro_batch()

        if batch is None:
            return None, None, None

        input_ids = torch.from_numpy(batch).to(torch.long).to(model_engine.device)
        labels = input_ids[..., :-2].contiguous()
        targets_1 = input_ids[..., 1:-1].contiguous()
        targets_2 = input_ids[..., 2:].contiguous()

        sum_tokens += torch.sum(targets_1 != -1).item()

        logits, loss = model_engine(labels, targets_1, targets_2)
        
        model_engine.backward(loss)
        model_engine.step()

        sum_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        sum_correct += torch.sum((predictions == targets_1) & (targets_1 != -1)).item()

    if dist.is_initialized():
        dist.all_reduce(torch.tensor([sum_loss, sum_tokens, sum_correct]).to(model_engine.device))

    return sum_loss / args.grad_accum, sum_tokens, sum_correct

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args, shard_config):
    logger.info(f"Node local_rank={args.local_rank} global_rank={args.global_rank} local_world_size={args.local_world_size} world_size={args.world_size}")

    if is_main_process:
        logger.info(f"Arguments: {args}")

    cfg = LatentLanguageConfig()
    cfg.n_vocab = round_up_to_next_multiple_of_128(shard_config["n_vocab"] + 1)
    cfg.padding_token = shard_config["n_vocab"]
    cfg.block_size = args.context
    
    model = LatentLanguage(cfg).cuda()

    model_engine, optimizer = setup_deepspeed(args, model)

    params = count_parameters(model)
    if is_main_process:
        logger.info(f"Total model parameters: {params/1000000.0:.2f}M")

    if args.verify_dataset:
        if is_main_process:
            logger.info("Verifying dataset...")
            is_valid = DataVerifier.verify(args.dataset_dir)
            if not is_valid:
                raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    if is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    dist.barrier()

    args.seed = synchronize_seed(args)

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
        _, client_state = model_engine.load_checkpoint(args.output_dir)
        if client_state is not None:
            step = client_state.get("step", 0)
            epoch = client_state.get("epoch", 0)
            total_steps = client_state.get("total_steps", 0)
            tokens = client_state.get("tokens", 0)
            wallclock_time = client_state.get("wallclock_time", 0.0)
            logger.info(f"Loaded checkpoint at step={step} epoch={epoch}")
        else:
            logger.info("No checkpoint found - Starting training from scratch")
    else:
        if args.local_rank == 0:
            logger.info("Resetting training - deleting output directory")
            recreate_folder(args.output_dir)
        dist.barrier()

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

            avg_train_loss, sum_tokens, sum_correct = train_one_step(
                args, model_engine, dataloader)

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

            if step % args.checkpoint_interval == 0:
                checkpoint_info = {
                    'train_version': 1,
                    'avg_train_loss': avg_train_loss,
                    'args': vars(args),
                    'total_steps': total_steps,
                    'epoch': epoch,
                    'step': step,
                    'tokens': tokens,
                    'wallclock_time': wallclock_time,
                    'lr': optimizer.param_groups[0]['lr'],
                }

                if args.wandb:
                    wandb.log(checkpoint_info)

                model_engine.save_checkpoint(args.output_dir, client_state=checkpoint_info)

        logger.info(f"Epoch {epoch} complete on rank {args.global_rank}")
        epoch += 1
        step = 0
        dist.barrier()

    logger.info(f"Training complete. Total steps: {total_steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output-dir", type=str, default="~/lllm_output", help="Path to model checkpoints for resuming") 
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random numbers. Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")

    parser.add_argument("--dataset-dir", type=str, default="~/dataset_shard", help="Dataset directory")
    parser.add_argument("--holdout-dir", type=str, default="~/holdout_shard", help="Holdout directory")
    parser.add_argument("--verify-dataset", action="store_true", help="Verify the dataset before training")
    parser.add_argument("--min-data-length", type=int, default=64, help="Minimum data length to use for training")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--project", type=str, default="my_project", help="Collection of experiments on wandb")

    parser.add_argument("--context", type=int, default=4098, help="Context size for each microbatch")
    parser.add_argument("--microbatch", type=int, default=1, help="Microbatch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.3, help="Weight decay for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")
    parser.add_argument("--steps", type=int, default=1000000, help="Total steps for training")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps (recommended 2000 above 100k steps)")
    parser.add_argument("--cooldown", type=int, default=200000, help="Cooldown steps (recommended 0.2x total steps)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient magnitude")

    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Steps between checkpoints")

    args = parser.parse_args()

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

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    main(args, shard_config)

    dist.destroy_process_group()
