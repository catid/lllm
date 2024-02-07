# Theory (1) Mamba and Transformer architectures are complementary:

# RNNs are designed for sequences and Transformers for ngram prediction.
# The best versions of these are excellent at both.

# "Mamba" (Gu and Dao, 2023)
# https://arxiv.org/pdf/2312.00752.pdf
# Figure 9 shows that adding attention layers improves performance.

# "Is Mamba Capable of ICL?" (Grazzi et al, 2024)
# https://arxiv.org/pdf/2402.03170.pdf
# In Figure 2 you can see that sometimes Transformer learns faster,
# and other times Mamba learns faster.
# They find that Mamba has a similar "iterative refinement" behavior.


# Theory (2) Transformer blocks appear to be forming functional pairs:

# "Subformer" (Reid et al, 2021)
# https://aclanthology.org/2021.findings-emnlp.344.pdf
# Shows that "every 2 layers shared" performs much better than all-shared,
# But not much better than less sharing.  So there's something important about pairs.

# "The Truth is in There" (LASER) (Sharma et al, 2023)
# https://arxiv.org/pdf/2312.13558.pdf
# Shows that every other layer seems to be low rank in a large language model,
# indicating that pairs of layers are plausible components.

# "PonderNet" (Banino et al, 2021)
# https://arxiv.org/pdf/2107.05407.pdf
# Shows that layers can be repeated and performance improves.
# There are previous results that indicate DNNs implement something like
# iterative optimizers, and these benefit from more iterations of course.
# But the "unit" of iteration seems to be pairs of blocks.


# Combining (1) and (2): Pairs of alternating Mamba/Transformer blocks
# is a powerful combination that seems to already benchmark well.
# Running the Transformer first makes sense to me since it grabs information
# from all over the input space and then Mamba can work on it a bit more locally.

import torch
from torch import nn, optim

from mamba_ssm import Mamba, Block
from transnormer import Transnormer
from model.linear_attention.srmsnorm import FastSimpleRMSNorm

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

class MambaFormer(nn.Module):
    def __init__(self, layer_index, args):
        super().__init__()

        self.transnormer = Transnormer(args)

        mamba_mixer = Mamba(
                    d_model=args.dim,
                    d_state=args.mamba_state,
                    d_conv=args.mamba_conv,
                    expand=args.mamba_expand)
        mamba_norm = FastSimpleRMSNorm(args.dim)

        self.mamba_block = Block(
            args.dim,
            mamba_mixer,
            norm_cls=mamba_norm,
            fused_add_norm=False,
            residual_in_fp32=False)

        _init_weights(self.transnormer, layer_index * 2)
        _init_weights(self.mamba_block, layer_index * 2 + 1)

    def forward(self, x):
        x = self.transnormer(x)

        # This does pre-norm inside mamba_block().
        # But they do not add the residual inside since we specify fused_add_norm=False
        hidden_states, _ = self.mamba_block(hidden_states=x, residual=None, inference_params=None)
        x += hidden_states

        return x
