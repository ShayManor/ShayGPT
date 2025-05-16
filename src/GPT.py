from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class GPTConfig(PretrainedConfig):
    def __init__(self, vocab_size, pad_id: Optional[int], n_layer=20, n_head=20, d_model=1280, dropout=0.06,
                 max_len=512, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(locals())


class GPT(PreTrainedModel):
    def __init__(self, cfg: GPTConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))

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
            "attn": nn.MultiheadAttention(d, h, dropout=drop, batch_first=True),
            "ln2": nn.LayerNorm(d),
            "mlp": nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
                nn.Dropout(drop)
            )
        })

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(
            self,
            input_ids=None,  # [B, T] token indices
            attention_mask=None,  # [B, T] 1 = keep, 0 = pad
            inputs_embeds=None,  # [B, T, d] optional pre-computed embeds
            labels=None,  # optional next-token labels
            use_cache=False,  # unused but accepted
            output_attentions=False,  # "
            output_hidden_states=False,  # "
            return_dict=False,  # "
            **kwargs  # catch anything new
    ):
        # ---------------- alias handling ----------------
        if inputs_embeds is not None:
            x = inputs_embeds  # skip token embed
        else:
            if input_ids is None:
                raise ValueError("Need input_ids or inputs_embeds")
            idx = input_ids
            x = self.tok_emb(idx) + self.pos_emb[:, : idx.size(1)]

        # pad-mask construction
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
        elif input_ids is not None:
            pad_mask = (input_ids == self.cfg.pad_id)
        else:
            pad_mask = None

        attn_mask = self.causal_mask[: x.size(1), : x.size(1)].to(x.device)

        # ---------------- transformer blocks ------------
        for blk in self.blocks:
            x = x + blk["attn"](
                blk["ln1"](x),
                blk["ln1"](x),
                blk["ln1"](x),
                attn_mask=attn_mask,
                key_padding_mask=pad_mask,
                need_weights=False  # we ignore attentions
            )[0]
            x = x + blk["mlp"](blk["ln2"](x))

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, V]

        if not return_dict:
            return logits
        else:  # mimic HF CausalLMOutputWithPast shape
            from transformers.modeling_outputs import CausalLMOutput
            return CausalLMOutput(
                loss=None,
                logits=logits,
                hidden_states=None if not output_hidden_states else (x,),
                attentions=None if not output_attentions else (None,),
            )
