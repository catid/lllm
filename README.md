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

# Install the nightly version of triton that includes parallel scan
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

# Run unit tests to make sure everything is working
pytest
```

## Train

```bash
./launch_local_train.sh
```
