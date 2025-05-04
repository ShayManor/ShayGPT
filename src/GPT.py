import torch
from torch import nn


class GPTConfig:
    def __init__(self,
                 vocab_size,
                 n_layer=6,
                 n_head=8,
                 d_model=512,
                 dropout=0.1,
                 max_len=1024):
        self.__dict__.update(locals())


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        block = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_head,
            dim_feedforward=4*cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True)
        self.blocks = nn.TransformerDecoder(block, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # weight tying → share weights with embedding
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self.register_buffer("causal_mask",
            torch.triu(torch.full((cfg.max_len, cfg.max_len), float("-inf")), 1))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx):          # idx: [B, T]
        B, T = idx.shape
        assert T <= self.pos_emb.size(1), "sequence too long"

        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        x = self.blocks(
            tgt=x,
            memory=None,             # no encoder memory in decoder‑only
            tgt_mask=self.causal_mask[:T, :T].to(x.device))
        x = self.ln_f(x)
        logits = self.lm_head(x)      # [B, T, V]
        return logits