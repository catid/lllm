# Ngram layer from "In-Context Language Learning: Architectures and Algorithms" (Akyurek, 2024)
# https://arxiv.org/pdf/2401.12973.pdf
# This reminds me of the LZ step of a general lossless data compressor.

# From seq_icl project:
# https://github.com/berlino/seq_icl/blob/77553249cc354480bd2375e8d3fca9784adc2c17/src/models/sequence/rnn/ngram.py#L45

import torch
from torch import nn
import torch.nn.functional as F

def induction_head(x, hidden_state, shift_step=1, ngram=1):
    _, seq_len = x.shape

    # bsz x L x L
    # match ngrams in the input sequence
    mask_0 = x[:, None, :] == x[:, :, None]

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=-1
    )
    mask_0 = torch.logical_and(mask_0, causal_mask)

    masks = [mask_0.long()]
    for _ in range(1, ngram):
        mask_0 = F.pad(mask_0, (1, -1, 1, -1), "constant", False)
        masks.append(mask_0.long())

    ih_mask = torch.stack(masks, dim=-1).sum(dim=-1) >= ngram

    if shift_step > 0:
        ih_mask = F.pad(ih_mask, (shift_step, -shift_step), "constant", False)

    ih_mask = torch.logical_and(ih_mask, causal_mask)

    ih_mask_norm = ih_mask / ih_mask.sum(dim=2, keepdim=True)
    ih_mask_norm = torch.nan_to_num(ih_mask_norm, 0)
    output = torch.einsum("bmn,bnz->bmz", ih_mask_norm, hidden_state)
    return output

class Ngram(nn.Module):
    def __init__(self, d_model, ngram=1):
        super().__init__()
        self.d_model = d_model
        self.ngram = ngram

        self.t0 = nn.Linear(self.d_model, self.d_model)
        self.t1 = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, input_ids=None):
        h0 = induction_head(input_ids, x, ngram=self.ngram)
        h1 = x
        y = self.t0(h0) + self.t1(h1)
        return y
