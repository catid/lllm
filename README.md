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
* Concatenate short strings together
* Add validation loss split to the sharding script, taking a random subset from each parquet file
* Windowed attention code: https://github.com/Dao-AILab/flash-attention/blob/0cb595ad943ac7539c49825f520659c0f61d4f40/flash_attn/bert_padding.py#L125

Training TODO:
* Get FSDP training working, and experiment with settings
* Switch to optimi https://optimi.benjaminwarner.dev/
* Experiment with StableAdamW, Lion, Kahan summation, gradient release, decouple_lr
* Incorporate cool stuff from Karpathy video + community (below)

Model TODO:
* Mamba2 interleaved with SWA layers
* SWA + Primer Spatial D-Conv 3x1: https://arxiv.org/pdf/2109.08668v2 (Figure 4)
* FFN layers: https://github.com/BlinkDL/RWKV-LM/blob/c2edfdc22a729b5f0cb896c99052c5c28902359a/RWKV-v5/src/model.py#L716
* But gate the FFN output per head rather than per float, and also gate the next layer per head via mask.
* Produce 4 tokens at once using 4x MLP heads

Karpathy video take-aways:
* Periodically print validation loss / example output
* 1.5M tokens/second training is the benchmark to beat
* torch.compile
* gradient accumulation up to 0.5M batch size, up to 1M for 1B parameter models
* nicer numbers (round out the vocab size)
* Use [Eluether harness to eval model](https://github.com/EleutherAI/lm-evaluation-harness)

```python
    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Further improvements from [community](https://github.com/KellerJordan/modded-nanogpt):
* 3x max learning rate
* Trapezoidal LR schedule: https://arxiv.org/pdf/2405.18392

Dataloader future improvements:
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation

Training future experiments:
* Keep training context size at 4096 tokens but vary the batch size during training.
* Gradient normalization: for p in model.parameters(): p.grad = p.grad / (p.grad.norm() + 1e-6)
* Can this replace layer normalization?

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
