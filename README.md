# WORK IN PROGRESS (do not use)

# Latent Large Language Models

A work in progress.

## (1) Node Setup

For a development setup with fast iterative deployment on LAN, follow the instruction from the `playbooks/` directory.

For Internet scale training, we will need to build a Docker container...


## (2) Dataset Setup

Follow the instructions in the `dataset/` directory.


## (3) Training

Follow the instructions in the `train/` directory.


# Ideas

* Take LLaMA-3 70B Instruct-tuned output from each data chunk, and train the model to generate the same continuations (a way to skip fine-tuning?)
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation
* Produce 4 tokens at once using 4x MLP heads
* VAR-style or diffusion style final block that predicts multiple tokens at once
* Byte model
* Spend a layer to drop tokens at KV cache point (mask for second half) per token
* Mixture of Depth (MoD)
* RWKV-6 for first half, Mix GLA and 20% Full attention for second half of YOCO
* https://github.com/sustcsonglin/flash-linear-attention for implementations of fast linear attention
* RWKV-6: https://github.com/Ronsor/rwkv-simple


* https://github.com/HazyResearch/ThunderKittens
* Sharing FFN weights onion-style https://arxiv.org/abs/2104.06022
* Share the majority of FFN weights between consecutive layers but only replace a few of them each time

* DeltaNet retrieval heads: https://github.com/sustcsonglin/flash-linear-attention/blob/4929ae7370f99ca60ab1d6e3903ba91ca9d981f0/fla/layers/delta_net.py#L48
https://arxiv.org/pdf/2406.06484

* Attention as a Hypernetwork: https://arxiv.org/pdf/2406.05816
https://github.com/smonsays/hypernetwork-attention

* https://github.com/Dao-AILab/flash-attention/blob/0cb595ad943ac7539c49825f520659c0f61d4f40/flash_attn/bert_padding.py#L125
