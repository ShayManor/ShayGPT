import math

import torch
import torch.nn as nn

from tokenizer import tokenizer


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
        seq_len = x.size(0)
        return x + self.pe[:seq_len].unsqueeze(1)


class TransformerEncoderModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            dimensions=512,
            nhead=8,
            nhid=2048,
            nlayers=6,
            dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            dimensions,
            padding_idx=tokenizer.token_to_id("[PAD]")
        )
        self.pos_encoder = PositionalEncoding(dimensions)
        encoder_layer = nn.TransformerEncoderLayer(dimensions, nhead, nhid, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.d_model = dimensions

    def forward(self, src_ids, src_mask):
        """
        :param src_ids: integer tensor [batch x seq_len]
        :param src_mask: [batch x seq_len x dimensions] floats
        :return: scale by \sqrt{dimensions}
        """
        x = self.embedding(src_ids).transpose(0, 1) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.encoder(x, src_key_padding_mask=~(src_mask.bool()))
        # output: (seq_len, batch, dimensions), positional encoding
        # Transpose to [batch x seq_len x dimensions]
        return output.transpose(0, 1)
