# Distributed Training

WIP

## Setup

This assumes the node is already provisioned by following instructions from the `playbooks/` directory.

Modify the `hosts.txt` file on your master node to match the ones you configured in the `dataset/` directory.

```bash
cd training
conda activate lllm

python make_train_script.py

./launch_train.sh
```

Or test locally:

```bash
cd training
conda activate lllm

torchrun --standalone --nproc_per_node=2 train.py
```
