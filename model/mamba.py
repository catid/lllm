# "Mamba" (Gu and Dao, 2023)
# https://arxiv.org/pdf/2312.00752.pdf
# Figure 9 shows that adding attention layers improves performance.

# "Is Mamba Capable of ICL?" (Grazzi et al, 2024)
# https://arxiv.org/pdf/2402.03170.pdf
# "Overall, our findings establish Mamba as an efficient and performant
# alternative to transformers for ICL involving longer input sequences."

# "Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks" (Park et al, 2024)
# https://arxiv.org/pdf/2402.04248.pdf
# They find that adding attention layers fixes decision tree,
# orthogonal-outlier regression, and vector-valued MQAR tests.

import torch
from torch import nn

from mamba_ssm import Mamba, Block
from model.srmsnorm import FastSimpleRMSNorm

import math

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MambaBlock(nn.Module):
    def __init__(self, layer_index, dim, state, conv, expand):
        super().__init__()

        mamba_mixer = Mamba(
                    d_model=dim,
                    d_state=state,
                    d_conv=conv,
                    expand=expand)
        mamba_norm = FastSimpleRMSNorm(dim)

        self.mamba_block = Block(
            dim,
            mamba_mixer,
            norm_cls=mamba_norm,
            fused_add_norm=False,
            residual_in_fp32=False)

        _init_weights(self.mamba_block, layer_index)

    def forward(self, x):
        # This does pre-norm inside mamba_block().
        # But they do not add the residual inside since we specify fused_add_norm=False
        hidden_states, _ = self.mamba_block(x, residual=None, inference_params=None)
        x += hidden_states

        return x
