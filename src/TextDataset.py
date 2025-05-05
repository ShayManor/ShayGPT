import itertools
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset

from tokenizer import tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, max_len=512):
        if isinstance(texts, str) and os.path.isfile(texts):
            with open(texts) as f:
                texts = [line.strip() for line in f if line.strip()]
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if self.pad_token_id is None:
            raise ValueError("Tokenizer has no [PAD] token; did you load the trained tokenizer?")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = tokenizer.encode(self.texts[idx]).ids[: self.max_len]
        ids += [self.pad_token_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class StreamDataset(IterableDataset):
    """
    Infinite/streaming dataset – wrap a HuggingFace streaming split.
    Each item is **raw text**; tokenisation happens in collate_batch.
    """

    def __init__(self, hf_stream, world_size, rank):
        super().__init__()
        self.hf_stream = hf_stream
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        for i, rec in enumerate(self.hf_stream):
            if i % self.world_size != self.rank:
                continue
            yield rec["text"]

    def __len__(self):
        return 100000  # arbitrary
