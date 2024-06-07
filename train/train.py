# Set the environment variables
import os

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from model.model import LatentLanguage, LatentLanguageConfig

import schedulefree # optimizer

import torch
import torch.nn as nn

import numpy as np
import random, time, json
import shutil
import argparse

import wandb

import deepspeed
from deepspeed import comm
from deepspeed import log_dist
from deepspeed.runtime.config import DeepSpeedConfig

from mora import MoRALayer, merge_mora_weights, replace_linear_with_mora

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

def synchronize_seed(local_rank, rank, seed=1337):
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

def delete_folder_contents(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def train_one_epoch(optimizer, lr_scheduler, criterion, model_engine, train_loader):
    model_engine.train()
    train_loss = 0.0
    num_batches = len(train_loader)

    for batch in train_loader:
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = input_ids.clone()

        outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()

    train_loss /= num_batches
    return train_loss

def validation_one_epoch(criterion, model_engine, val_loader):
    model_engine.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = input_ids.clone()

            outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            val_loss += loss.item()

            _, predicted = shift_logits.max(dim=-1)
            total += shift_labels.numel()
            correct += predicted.eq(shift_labels).sum().item()

    val_loss /= num_batches
    accuracy = 100.0 * correct / total
    return val_loss, correct, total, accuracy

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

def main(args):
    t0 = time.time()

    # Initialize DeepSpeed
    deepspeed.init_distributed(
        dist_backend="nccl",
        verbose="false"
    )

    cfg = LatentLanguageConfig()
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
        lr_scheduler=None, # ScheduleFree does not support LR schedulers
        config=ds_config,
        model_parameters=model.parameters())

    log_0(f"Arguments: {args}")

    comm.barrier()

    fp16 = model_engine.fp16_enabled()
    log_0(f'model_engine.fp16_enabled={fp16}')

    rank = model_engine.local_rank
    shard_id = model_engine.global_rank
    num_gpus = model_engine.world_size
    train_batch_size = model_engine.train_batch_size()
    data_loader_batch_size = model_engine.train_micro_batch_size_per_gpu()
    steps_per_print = model_engine.steps_per_print()

    seed = synchronize_seed(args, rank, shard_id)

    log_all(f"Node rank={rank}, num_shards={num_gpus}, shard_id={shard_id}, train_batch_size={train_batch_size}, data_loader_batch_size={data_loader_batch_size}, steps_per_print={steps_per_print}, seed={seed}")  

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

    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    avg_val_loss = float("inf")
    start_epoch = 0
    end_epoch = 0
    epochs_without_improvement = 0

    if args.reset:
        log_0("Resetting training - deleting output directory")
        if rank == 0:
            delete_folder_contents(args.output_dir)
        comm.barrier()
    else:
        _, client_state = model_engine.load_checkpoint(load_dir=args.output_dir)
        if client_state is not None:
            start_epoch = client_state['epoch'] + 1
            avg_val_loss = client_state['avg_val_loss']
            best_val_loss = avg_val_loss
            log_all(f"Loaded checkpoint at epoch {client_state['epoch']}")
        else:
            log_all("No checkpoint found - Starting training from scratch")

    # Load the dataset
    dataset = load_dataset("allenai/dolma", split="train", name="v1_6-sample")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt", max_length=1024)

    split_dataset = split_dataset_by_node(dataset, rank=shard_id, world_size=num_gpus)

    tokenized_dataset = split_dataset.map(tokenize_function, batched=True, num_proc=24, remove_columns=["text"])

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_dataset, batch_size=data_loader_batch_size, collate_fn=data_collator, shuffle=True
    )

    for epoch in range(start_epoch, args.max_epochs):
        end_epoch = epoch
        start_time = time.time()

        train_loss = train_one_epoch(optimizer, criterion, model_engine, tokenized_dataset)

        val_loss, correct, total, examples = validation_one_epoch(criterion, model_engine, tokenized_dataset)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Sync variables between machines
        sum_train_loss = torch.tensor(train_loss).cuda(rank)
        sum_val_loss = torch.tensor(val_loss).cuda(rank)
        sum_correct = torch.tensor(correct).cuda(rank)
        sum_total = torch.tensor(total).cuda(rank)
        comm.all_reduce(tensor=sum_train_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_val_loss, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_correct, op=comm.ReduceOp.SUM)
        comm.all_reduce(tensor=sum_total, op=comm.ReduceOp.SUM)

        total_train_items = len(train_loader) * num_gpus
        total_val_items = len(val_loader) * num_gpus
        comm.barrier()
        avg_train_loss = sum_train_loss.item() / total_train_items
        avg_val_loss = sum_val_loss.item() / total_val_items
        val_acc = 100. * sum_correct / sum_total

        if is_main_process():
            log_0(f"Epoch {epoch + 1} - TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.2f}%, Time={epoch_time:.2f} sec")

            if args.wandb:
                lr = optimizer.param_groups[0]['lr']
                wandb.log({"avg_train_loss": avg_train_loss, "val_acc": val_acc, "avg_val_loss": avg_val_loss, "epoch": epoch, "wallclock_time": epoch_time, "lr": lr})

        # Check if validation loss has improved
        if val_acc > best_val_acc:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc
            best_train_loss = avg_train_loss
            epochs_without_improvement = 0

            log_0(f'New best validation loss: {best_val_loss:.4f}  Validation accuracy: {best_val_acc:.2f}%')

            client_state = {
                'train_version': 1,
                'avg_train_loss': avg_train_loss,
                'avg_val_loss': avg_val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
                'fp16': fp16,
            }
            model_engine.save_checkpoint(save_dir=args.output_dir, client_state=client_state)

            if is_main_process():
                save_deepspeed_model_engine(model_engine, fp16, args)
                log_0(f"Wrote model to {args.output_model} with val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= args.patience:
                log_0(f"Early stopping at epoch {epoch} due to epochs_without_improvement={epochs_without_improvement}")
                break

    if is_main_process():
        t1 = time.time()
        dt = t1 - t0

        log_0(f'Training complete after {dt} seconds.  Best model was written to {args.output_model}  Final best validation loss: {best_val_loss}, best validation accuracy: {best_val_acc:.2f}%')

        if args.wandb:
            wandb.log({"best_val_loss": best_val_loss, "best_val_acc": best_val_acc})
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")

    # Deepspeed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Path to the latest checkpoint for resuming")
    parser.add_argument("--reset", action="store_true", help="Reset training from scratch")
    parser.add_argument("--output-model", type=str, default="cifar10.pth", help="Output model file name")

    # Misc
    parser.add_argument("--seed", type=int, default=-1, help="Seed for random numbers.  Set to -1 to pick a fully random seed")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--fp32", action='store_true', help="Enable fp32 training (fp16 default)")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--name", type=str, default="", help="Give your experiment a name")
    parser.add_argument("--project", type=str, default="my_project", help="Collection of experiments on wandb")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay for training")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for training")

    # Training duration
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=50, help="Patience for validation loss not decreasing before early stopping")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if args.deepspeed_config==None or len(args.deepspeed_config)==0:
        args.deepspeed_config = "deepspeed_config.json"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
