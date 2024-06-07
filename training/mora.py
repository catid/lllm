import torch
import torch.nn as nn
import torch.nn.functional as F

class MoRALayer(nn.Module):
    def __init__(self, d_in, d_out, mora_mult=2, weight=None, bias=None):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias) # trainable
        else:
            self.bias = None

        # ReMoRA parameters: https://arxiv.org/pdf/2405.12130
        assert d_in % mora_mult == 0, f"d_in={d_in} must be divisible by mora_mult={mora_mult}"
        assert d_out % mora_mult == 0, f"d_out={d_out} must be divisible by mora_mult={mora_mult}"
        self.mora_mult = mora_mult
        self.m_in = d_in // mora_mult
        self.m_out = d_out // mora_mult
        self.mora = torch.nn.Parameter(torch.zeros(self.m_out, self.m_in))
        self.compress_type = 1

        # DoRA parameters: https://arxiv.org/pdf/2402.09353
        self.dora_mag = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

        self.mora_acc = [
            nn.Parameter(torch.zeros(self.m_out, self.m_in), requires_grad=False),
            nn.Parameter(torch.zeros(self.m_out, self.m_in), requires_grad=False),
        ]

    def merge(self):
        with torch.no_grad():
            # Accumulate the updates to original weights.
            # These can be transmitted along with dora_mag to compress weight updates.
            self.mora_acc[self.compress_type] += self.mora

            if self.compress_type == 0:
                w = self.mora.repeat(self.mora_mult, self.mora_mult)
            else:
                w = self.mora.repeat_interleave(self.mora_mult, dim=0).repeat_interleave(self.mora_mult, dim=1)

            self.weight += w

            self.mora.zero_()

            self.compress_type ^= 1

    def forward(self, x):
        w = self.weight

        # ReMoRA
        if self.compress_type == 0:
            w = w + self.mora.repeat(self.mora_mult, self.mora_mult)
        else:
            w = w + self.mora.repeat_interleave(self.mora_mult, dim=0).repeat_interleave(self.mora_mult, dim=1)

        # DoRA
        w = self.dora_mag * w / w.norm(p=2, dim=0, keepdim=True)

        return F.linear(x, w, self.bias)

def replace_linear_with_mora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Get the input and output dimensions of the current nn.Linear layer
            d_in = module.in_features
            d_out = module.out_features

            # Handle optional bias
            bias = None
            if hasattr(module, 'bias'):
                bias = module.bias.data.clone()

            # Create a new DoRALayer with the same dimensions
            r = MoRALayer(
                d_out=d_out,
                d_in=d_in,
                weight=module.weight.data.clone(),
                bias=bias)
            setattr(model, name, r)
        else:
            # Recursively apply this function to submodules
            replace_linear_with_mora(module)

def merge_mora_weights(model):
    for _, module in model.named_children():
        if isinstance(module, MoRALayer):
            module.merge()
        else:
            # Recursively apply this function to submodules
            merge_mora_weights(module)
