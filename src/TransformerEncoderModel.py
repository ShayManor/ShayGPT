import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        sin_table = torch.zeros(max_len, embedding_dim)
        # column vector [0,1,2,…,max_len-1]^T, shape [max_len × 1], cast to float, represents each position index.
        pos_index = torch.arange(0, max_len).unsqueeze(1).float()
        # Compute inverse wavelengths for sin/cos terms and get frequency for each pair of dimensions
        div = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        # Even indices are sin, odd indices are cos
        sin_table[:, 0::2] = torch.sin(pos_index * div)
        sin_table[:, 1::2] = torch.cos(pos_index * div)
        self.register_buffer("pe", sin_table)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[: x.size(0)]


