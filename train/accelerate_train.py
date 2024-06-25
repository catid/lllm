import os, random, time, shutil, argparse, yaml, math, copy, sys
from packaging import version

import torch
from torch.nn.parallel import DistributedDataParallel
from functools import partial

from model.model import LatentLanguage, LatentLanguageConfig, MultiQueryAttentionDConv

import numpy as np

import wandb

from cpp_dataloader import DataLoader, DataVerifier, EpochConfig

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from accelerate.optimizer import AdaLOMO

from logger_tt import setup_logging, logger

def get_current_script_directory():
    script_path = os.path.abspath(sys.argv[0])
    script_dir = os.path.dirname(script_path)
    return script_dir

setup_logging(
    use_multiprocessing="fork",
    log_path=os.path.join(get_current_script_directory(), "train.log"))

def round_up_to_next_multiple_of_128(x):
    if x % 128 == 0:
        return x
    return x + (128 - (x % 128))

def get_true_random_32bit_positive_integer():
    random_bytes = bytearray(os.urandom(4))
    random_bytes[0] &= 0x7F
    random_int = int.from_bytes(bytes(random_bytes), byteorder='big')
    return random_int

def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def get_lr(args, step):
    assert step <= args.steps
    if step < args.warmup:
        return args.lr * (step+1) / args.warmup
    elif step < args.steps - args.cooldown:
        return args.lr
    else:
        decay_ratio = (step - (args.steps - args.cooldown)) / args.cooldown
        return args.lr * (1 - math.sqrt(decay_ratio))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_accelerator(args):
    return Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
    )

def train_one_step(args, accelerator, model, dataloader):
    model.train()

    sum_loss = 0.0
    sum_tokens = 0
    sum_correct = 0

    for grad_accum_step in range(args.grad_accum):
        batch, is_cont, step, total_steps = dataloader.get_micro_batch()

        if batch is None:
            return None, None, None

        input_ids = torch.from_numpy(batch).to(torch.long)
        labels = input_ids[..., :-2].contiguous()
        targets_1 = input_ids[..., 1:-1].contiguous()
        targets_2 = input_ids[..., 2:].contiguous()

        sum_tokens += torch.sum(targets_1 != -1).item()

        with accelerator.autocast():
            logits, loss = model(labels, targets_1, targets_2)
            loss = loss / args.grad_accum

        accelerator.backward(loss)

        sum_loss += loss.detach().item()

        predictions = torch.argmax(logits, dim=-1)
        sum_correct += torch.sum((predictions == targets_1) & (targets_1 != -1)).item()

    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

    lr = get_lr(args, step)
    for param_group in accelerator.optimizer.param_groups:
        param_group['lr'] = lr

    accelerator.optimizer.step()
    accelerator.optimizer.zero_grad(set_to_none=True)

    return sum_loss, sum_tokens, sum_correct

def main(args, shard_config):
    accelerator = setup_accelerator(args)
    device = accelerator.device

    logger.info(f"Process rank: {accelerator.process_index}, total_processes: {accelerator.num_processes}")
    logger.info(f"Distributed type: {accelerator.distributed_type}")

    if accelerator.is_main_process:
        logger.info(f"Arguments: {args}")

    cfg = LatentLanguageConfig()
    cfg.n_vocab = round_up_to_next_multiple_of_128(shard_config["n_vocab"] + 1)
    cfg.padding_token = shard_config["n_vocab"]
    cfg.block_size = args.context
    model = LatentLanguage(cfg).to(device)

    # Use Accelerate's AdaLOMO optimizer
    optimizer = AdaLOMO(model.parameters(), lr=args.lr)

    model, optimizer = accelerator.prepare(model, optimizer)

    params = count_parameters(model)
    if accelerator.is_main_process:
        logger.info(f"Total model parameters: {params/1000000.0:.2f}M")

    if args.verify_dataset and accelerator.is_main_process:
        logger.info("Verifying dataset...")
        is_valid = DataVerifier.verify(args.dataset_dir)
        if not is_valid:
            raise RuntimeError("Dataset is corrupted and must be regenerated using dataset/shard_dataset.py")

    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    accelerator.wait_for_everyone()

    set_seed(args.seed)

    if args.wandb and accelerator.is_main_process:
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
        checkpoint_info = accelerator.load_state(args.output_dir)
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
        if accelerator.is_main_process:
            logger.info("Resetting training - deleting output directory")
            recreate_folder(args.output_dir)
        accelerator.wait_for_everyone()

    dataloader = DataLoader(args.dataset_dir)
    validation_dataloader = DataLoader(args.holdout_dir)

    while total_steps < args.steps:
        config = EpochConfig()
        config.seed0 = args.seed
        config.seed1 = epoch
        config.local_rank = accelerator.process_index
        config.local_rank_count = accelerator.num_processes
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
                args, accelerator, model, dataloader)

            if avg_train_loss is None:
                logger.info(f"Epoch {epoch} data exhausted at step={step}")
                break

            end_time = time.time()
            step_time = end_time - start_time

            tokens += sum_tokens
            tokens_per_second = sum_tokens / step_time
            correct_pct = sum_correct * 100.0 / sum_tokens

            if accelerator.is_main_process:
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

                if args.wandb and accelerator.is_main_process:
                    wandb.log(checkpoint_info)

                accelerator.save_state(args.output_dir)

        logger.info(f"Epoch {epoch} complete")
        epoch += 1
        step = 0
        accelerator.wait_for_everyone()

    logger.info(f"Training complete. Total steps: {total_steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--output-dir", type=str, default="~/lllm_output", help="Path to model checkpoints for resuming") 
    parser.add_argument("--resume", action="store_true", help="Resume training from previous checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random numbers. Set to -1 to pick a fully random seed")
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")
    parser.add_argument("--steps", type=int, default=1000000, help="Total steps for training")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps (recommended 2000 above 100k steps)")
    parser.add_argument("--cooldown", type=int, default=200000, help="Cooldown steps (recommended 0.2x total steps)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient magnitude")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Steps between checkpoints")

    args = parser.parse_args()

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

    logger.info(f"Shard config: {shard_config}")

    main(args, shard_config)
