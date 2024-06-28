# WORK IN PROGRESS (do not use)

# Latent Large Language Models

A work in progress.

## (1) Node Setup

For a development setup with fast iterative deployment on LAN, follow the instruction from the `playbooks/` directory.

For Internet scale training, we will need to build a Docker container...


## (2) Dataset Setup

Follow the instructions in the `dataset/` directory.


## (3) Training

```bash
conda create -n lllm python=3.10 -y
conda activate lllm
pip install "numpy<2.0"
pip install packaging torch==2.3.1 torchvision torchaudio
pip install mamba_ssm
pip install causal-conv1d
pip install flash-attn
pip install -r requirements.txt
pip install cupy
```

Follow the instructions in the `train/` directory.


# TODO

Training TODO:
* Implement Async DiLoCo: https://arxiv.org/pdf/2401.09135v1

Dataloader TODO:
* Add support for returning the list of concatenated samples in flash_attn format
* Add support for DCLM-Baseline 4T: https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0

Dataloader future improvements:
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation

Training future experiments:
* Meta-learning to try to estimate weight updates using larger batch sizes and more iterations from smaller batch sizes and single steps
* Try optimi https://optimi.benjaminwarner.dev/

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
* VQVAE compression of embedding layers: https://arxiv.org/abs/2406.11837
* SparseK KV cache compression: https://arxiv.org/pdf/2406.16747

Fine-tuning ideas:
* Take LLaMA-3 70B Instruct-tuned output from each data chunk, and train the model to generate the same continuations (a way to skip fine-tuning?)

Onion training:

1) Start with a very small model that is: nn.Embed -> SambaBlock1 -> Quantized1 (8-bit) -> SambaBlock1 -> heads.  nn.Embed is taken from a pre-trained large model and is frozen.  SambaBlock1 blocks have shared parameters.  There is a FFN head that reproduces the input token ids with reconstruction loss.  There is a second FFN head that predicts the next token with cross-entropy loss.  And a third head that predicts the following token.
(2) Train the model so that loss = reconstruction + next_token + second_next_token until it converges.
(3) Freeze SambaBlock1.  Insert a new SambaBlock2:
nn.Embed -> SambaBlock1 -> Quantized1 -> SambaBlock2 -> Quantized2 -> SambaBlock2 -> Quantized1 -> SambaBlock1 -> heads
(4) Continue training until convergence.
(5) Repeat with a third block, etc.
The Quantized layer involves kind of an auto-encoder thing that you split in half when inserting more blocks in between
