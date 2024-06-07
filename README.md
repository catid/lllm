# WORK IN PROGRESS (do not use)

# Latent Large Language Models

A work in progress.

## (1) Node Setup

For a development setup with fast iterative deployment on LAN, follow the instruction from the `playbooks/` directory.

For Internet scale training, we will need to build a Docker container...


## (2) Dataset Setup

Follow the instructions in the `dataset/` directory.


## (3) Training

Follow the instructions in the `training/` directory.


# Ideas

* ReMoRa with Sophia optimizer https://github.com/Liuhong99/Sophia/blob/main/sophia.py
* Speedup https://github.com/Liuhong99/Sophia/pull/35

* Non-causal prompt processing for first half of the sequence, causal for the second half - This would allow us to do some kind of rolling context compression for long sequences
* Take LLaMA-3 70B Instruct-tuned output from each data chunk, and train the model to generate the same continuations (a way to skip fine-tuning?)
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation
* Produce 4 tokens at once using 4x MLP heads
* VAR-style or diffusion style final block that predicts multiple tokens at once
* Byte model
* Share the majority of FFN weights between consecutive layers but only replace a few of them each time
* Spend a layer to drop tokens at KV cache point (mask for second half) per token
* Mixture of Depth (MoD)
* RWKV-6 for first half, Mix GLA and 20% Full attention for second half of YOCO
* https://github.com/sustcsonglin/flash-linear-attention for implementations of fast linear attention
* RWKV-6: https://github.com/Ronsor/rwkv-simple


FP16 training performance/watt:
H100: 989 TFLOPS / 1275W = 0.78 TFLOPS/W
B200: 2250 TFLOPS / 1788W = 1.26 TFLOPS/W
Improvement in one year due to Nvidia hardware improvements: 60% better efficiency.

Source: https://www.semianalysis.com/p/nvidia-blackwell-perf-tco-analysis
