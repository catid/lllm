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

Dataloader TODO:
* Add support for returning the list of concatenated samples in flash_attn format
* Add support for DCLM-Baseline 4T: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0

Dataloader future improvements:
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation

Training future experiments:
* Meta-learning to try to estimate weight updates using larger batch sizes and more iterations from smaller batch sizes and single steps
* Try optimi https://optimi.benjaminwarner.dev/
* AdaLOMO

FFN experiments:
* Sharing FFN weights onion-style https://arxiv.org/abs/2104.06022
* Share the majority of FFN weights between consecutive layers but only replace a few of them each time

Future model experiments:
* Use an FFN output from previous layer to select which tokens are masked in the next SWA layer (MoD)
* Instead of gating the whole layer, gate each head.  Same for FFN output head gating.
* VAR-style or diffusion style final block that predicts multiple tokens at once
* Byte model
* Spend a layer to drop tokens at KV cache point (mask for second half) per token
* RWKV-6 for first half, Mix GLA and 20% Full attention for second half of YOCO
* https://github.com/sustcsonglin/flash-linear-attention for implementations of fast linear attention
* RWKV-6: https://github.com/Ronsor/rwkv-simple
* https://github.com/HazyResearch/ThunderKittens
* DeltaNet retrieval heads: https://github.com/sustcsonglin/flash-linear-attention/blob/4929ae7370f99ca60ab1d6e3903ba91ca9d981f0/fla/layers/delta_net.py#L48
https://arxiv.org/pdf/2406.06484
* Mamba2 with cu_seqlen: https://github.com/zigzagcai/mamba
Needs some fixes from master mamba branch
* Mamba2 h_t ring buffer: https://x.com/MrCatid/status/1800995821543907626
* Samba: SWA cross-attend to much earlier tokens deeper in to avoid squashed representation collapse

Fine-tuning ideas:
* Take LLaMA-3 70B Instruct-tuned output from each data chunk, and train the model to generate the same continuations (a way to skip fine-tuning?)
