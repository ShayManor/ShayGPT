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

    def __init__(self, hf_stream):
        super().__init__()
        self.stream = hf_stream

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            it = self.stream
        else:
            it = itertools.islice(self.stream, worker_info.id, None, worker_info.num_workers)
        for record in it:
            yield record["text"]
