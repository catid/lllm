# Latent Large Language Models

Latent LLMs add downsamplers to the input and upsamplers to the output, allowing the majority of the model to operate in a smaller latent space.

## Setup

Tested hardware configurations:

* 4x Ubuntu 23.10 Linux Servers each with 2x Nvidia RTX 4090 GPUs.

Requires conda: https://docs.conda.io/projects/miniconda/en/latest/

```
git clone https://github.com/catid/lllm
cd lllm

conda create -n lllm python=3.10 -y && conda activate lllm

pip install -r requirements.txt
```
