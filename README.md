**Small GPT-Style Transformer for Next-Token Prediction**

## Overview

This is a lightweight, end-to-end implementation of a GPT-style Transformer tailored for next-token prediction on English text. It leverages a curated 1 billion-token RedPajama corpus sample and custom data cleaning to train a model with \~45 million parameters. The system incorporates modern best practices—mixed precision, GPU acceleration, and PyTorch 2.0 compilation—for efficient training and seamless inference.

## Features

* **Data preprocessing**: Stream, clean, and tokenize `.jsonl` shards or CSV/JSON files via `full_clean` and HuggingFace Tokenizers (BPE).
* **Tokenizer**: Sub-word Byte-Pair Encoding with special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `[BOS]`), padding, and truncation.
* **Model architecture**: GPT-style decoder stack implemented in `GPT.py` (configurable depth, heads, and hidden sizes).
* **Training optimizations**:

  * Automatic Mixed Precision (AMP) using `torch.cuda.amp`
  * PyTorch 2.0 `torch.compile` for kernel fusion
  * Pinned-memory `DataLoader` with multiple workers
  * Cosine annealing learning-rate scheduler and AdamW optimizer
* **Inference**: Greedy or multinomial next-token generation from arbitrary context.

## Repository Structure

```
workspace/
├── data/                  # Local CSV or JSON text data
├── redpajama-1B/         # RedPajama sample shards (.jsonl)
├── src/
│   ├── clean.py          # full_clean: streaming reader & text normalization
│   ├── tokenizer.py      # BPE tokenizer training & loading
│   ├── TextDataset.py    # PyTorch Dataset for token IDs + attention masks
│   ├── TransformerEncoderModel.py
│   ├── GPT.py            # GPTConfig & GPT model class
│   ├── train.py          # Training & inference entry point
│   └── utils/            # Optional helper modules
└── README.md             # This file
```

## Prerequisites

* Python 3.10+
* PyTorch 2.0+ with CUDA support
* `tokenizers`, `datasets`, `pandas`, `matplotlib`

Install via:

```bash
python -m venv venv && source venv/bin/activate
pip install torch tokenizers datasets pandas matplotlib
```

## Data Preparation

1. **Download RedPajama shards** into `src/redpajama-1B/`:

   * `c4_sample.jsonl` (866 MB)
   * `cc_2023-06_sample.jsonl` (857 MB)
   * `cc_2020-05_sample.jsonl` (836 MB)
2. **Optionally add** your own CSV/JSON via `data/AI_Human.csv`

The `full_clean(paths)` function in `src/clean.py` accepts a list of file paths (`.jsonl`, `.json`, or `.csv`) and returns a list of cleaned text strings.

## Tokenizer Training & Loading

The first run of `src/tokenizer.py`:

* Trains a BPE tokenizer on the provided shards
* Saves to `src/tokenizer.json`
* Enables padding and truncation

Subsequent imports simply load the existing tokenizer:

```python
from src.tokenizer import tokenizer
vocab_size = tokenizer.get_vocab_size()
```

## Training

Launch training with:

```bash
python src/train.py
```

Key hyperparameters live in `train()`:

* `shards`: list of three `.jsonl` files
* `epochs`: number of passes over data
* `init_batch`: starting batch size (auto‐adjust if OOM)
* `lr`: learning rate for AdamW

The script:

1. Builds `TextDataset` (max\_len = 256 tokens)
2. Creates `DataLoader` (pinned memory, multi‐worker)
3. Instantiates `GPT` model & moves to CUDA
4. Wraps forward/backward in `torch.cuda.amp.autocast` + `GradScaler`
5. Steps `optimizer` + `scheduler` each batch
6. Saves checkpoints per epoch (`transformer_encoder_e{epoch}.pth`)

## Inference

```python
import torch
from src.tokenizer import tokenizer
from src.GPT import GPTConfig, GPT

# Load model
cfg   = GPTConfig(vocab_size=tokenizer.get_vocab_size())
model = GPT(cfg).to("cuda")
model.load_state_dict(torch.load("transformer_encoder_e9.pth"))
model.eval()

# Tokenize prompt
text    = "the dog is feeling "
enc     = tokenizer.encode(text)
ids     = torch.tensor([enc.ids], device="cuda")  # [1, L]

# Forward pass with no grad
with torch.no_grad():
    logits = model(ids)                   # [1, L, V]
last_logits = logits[0, -1]
probs       = torch.softmax(last_logits, dim=-1)
next_id     = torch.argmax(probs).item()
next_token  = tokenizer.id_to_token(next_id)
print("Next word →", next_token)
```

Loop the above to generate multi‐token continuations, or use sampling via `torch.multinomial(probs, k)`.

## How It Works (High Level)

1. **Tokenize**: raw text → sub-word IDs + attention masks
2. **Embed**: integer IDs → dense vectors via `nn.Embedding`
3. **Add Positional Encoding**: sinusoidal signals for order
4. **Stack Decoder Layers**: multi-head self-attention + feed-forward + residuals
5. **Project**: hidden states → vocabulary logits via tied weights
6. **Compute Loss**: Cross-entropy next-token, ignoring `[PAD]`
7. **Backprop & Update**: AMP for speed, scheduler for LR decay
8. **Inference**: feed prompt, sample or greedy decode next token

## Performance Tips

* Use the **AMP + GradScaler** snippet in `train.py` to cut memory and speed
* Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
* Run on a modern GPU (A100/V100) with large `init_batch`
* Monitor GPU utilization via `nvidia-smi` or vast.ai dashboard

---

run with
```commandline
torchrun --nproc_per_node=4 train.py --config gpt_d768_l12.yml
```
torchrun --nproc_per_node=1 train.py --epochs=50
For questions or contributions, open an issue or submit a pull request. Stay innovative!
© 2025 Shay Manor
