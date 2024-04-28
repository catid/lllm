# WORK IN PROGRESS (do not use)

# Latent Large Language Models

Latent LLMs add downsamplers to the input and upsamplers to the output, allowing the majority of the model to operate in a smaller latent space.

We choose a 4:1 downsampling/upsampling ratio, so the model produces 4 tokens at a time and must pad input up to be a multiple of 4 tokens.

## Setup

Tested hardware configurations:

* 4x Ubuntu 23.10 Linux Servers each with 2x Nvidia RTX 4090 GPUs.

Requires conda: https://docs.conda.io/projects/miniconda/en/latest/

```
git clone https://github.com/catid/lllm
cd lllm

conda create -n lllm python=3.10 -y && conda activate lllm

pip install -r requirements.txt

# Run unit tests to make sure everything is working
pytest

# Download the dataset
python -m dataset.download_dataset
```

## Train

```bash
./launch_local_train.sh
```

# Ideas

* Non-causal prompt processing for first half of the sequence, causal for the second half - This would allow us to do some kind of rolling context compression for long sequences
* Take LLaMA-3 70B Instruct-tuned output from each data chunk, and train the model to generate the same continuations (a way to skip fine-tuning?)
* RHO-loss for the dataset using LLaMA-3 8B to provide reference loss for each token - need to convert to our tokenizer via approximation
* Produce 4 tokens at once
* Train on 8K context
* 90% RWKV-6, 10% Full Attention similar to Infini-Attention
* MambaByte/SpaceByte style tokenization instead of using RWKV tokenizer
