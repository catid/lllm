import torch
import torch.nn as nn
import torch.nn.functional as F

class MoRALayer(nn.Module):
    def __init__(self, d_in, d_out, mora_div=2, weight=None, bias=None):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.Tensor(d_out, d_in), requires_grad=False)

        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

        assert d_in % mora_div == 0, f"d_in={d_in} must be divisible by mora_div={mora_div}"
        assert d_out % mora_div == 0, f"d_out={d_out} must be divisible by mora_div={mora_div}"
        self.mora_div = mora_div
        self.m_in = d_in // mora_div
        self.m_out = d_out // mora_div

        self.mora = torch.nn.Parameter(torch.zeros(self.m_in, self.m_out))

        self.dora_mag = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

        self.compress_type = 1

    def merge(self):
        with torch.no_grad():
            if self.compress_type == 0:
                w = self.mora.repeat(self.mora_div, self.mora_div)
            else:
                w = self.mora.repeat_interleave(self.mora_div, dim=0).repeat_interleave(self.mora_div, dim=1)

            self.weight += w

            self.mora.zero_()

            self.compress_type ^= 1

    def forward(self, x):
        w = self.weight

        if self.compress_type == 0:
            w = w + self.mora.repeat(self.mora_div, self.mora_div)
        else:
            w = w + self.mora.repeat_interleave(self.mora_div, dim=0).repeat_interleave(self.mora_div, dim=1)

        norm_adapted = w / w.norm(p=2, dim=0, keepdim=True)
        w = self.dora_mag * norm_adapted

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

def merge_weights(model):
    for _, module in model.named_children():
        if isinstance(module, MoRALayer):
            module.merge()
        else:
            # Recursively apply this function to submodules
            merge_weights(module)
