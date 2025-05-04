import torch
from torch import nn
from flash_attn.modules.mha import MHA
from flash_attn.rotary import apply_rotary_emb, RotaryEmbedding

class GPTConfig:
    def __init__(self,
                 vocab_size,
                 n_layer=18,
                 n_head=18,
                 d_model=1224,
                 dropout=0.1,
                 max_len=512):
        self.__dict__.update(locals())


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rotary = RotaryEmbedding(cfg.d_model // cfg.n_head)

        self.blocks = nn.ModuleList([
            self._build_block() for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.max_len, cfg.max_len) * float("-inf"), 1)
        )
        self.apply(self._init_weights)

    def _build_block(self):
        d, h, drop = self.cfg.d_model, self.cfg.n_head, self.cfg.dropout
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(d),
            "attn": MHA(d, h, dropout=drop, causal=True, bias=False),  # flash‑attn
            "ln2": nn.LayerNorm(d),
            "mlp": nn.Sequential(
                nn.Linear(d, 4*d, bias=False),
                nn.GELU(),
                nn.Linear(4*d, d, bias=False),
                nn.Dropout(drop)
            )
        })

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):        # idx: [B,T]
        B, T = idx.shape
        assert T <= self.cfg.max_len
        x = self.tok_emb(idx * (self.cfg.d_model ** 0.5))
        for blk in self.blocks:
            x = x + blk["attn"](blk["ln1"](x))
            x = x + blk["mlp"](blk["ln2"](x))
        x = self.ln_f(x)
        return self.lm_head(x)