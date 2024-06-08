# Distributed Training

To reduce VRAM and IO requirements for training, we implement the ReMoRa training algorithm from "MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning" paper ( https://arxiv.org/abs/2405.12130 ).

Since each node often has multiple GPUs, we use deepspeed to train on multiple GPUs in parallel.  Updates between nodes are shared during our own RPC protocol.


## Setup

This assumes the node is already provisioned by following instructions from the `playbooks/` directory.

Modify the `hosts.txt` file on your master node to match the ones you configured in the `dataset/` directory.

```bash
cd training
conda activate lllm

# Verify that training works
./launch_local_train.sh --reset
```
