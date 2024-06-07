import torch

class MoraLinear:
    def __init__(self, d_in, d_out, mora_mult=4, bias=True):
        self.d_in = d_in
        self.d_out = d_out
        self.bias = bias
        self.w = torch.nn.Parameter(torch.randn(d_in, d_out))
        if bias:
            self.b = torch.nn.Parameter(torch.randn(d_out))

        # 
        mora_in = (d_in + mora_mult - 1) // mora_mult

        self.m = 

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b
