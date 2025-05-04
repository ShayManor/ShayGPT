import numpy as np
import pandas as pd
import torch
from torch import float32
from torch.utils.data import Dataset

from src.clean import full_clean
from src.tokenizer import tokenizer


class TextDataset(Dataset):
    def __init__(self, path, max_len=128):
        self.texts = full_clean(path)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if self.pad_token_id is None:
            raise ValueError("Tokenizer has no [PAD] token; did you load the trained tokenizer?")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer.encode(self.texts[idx])
        ids = list(encoded.ids)[: self.max_len]
        att = [1] * len(ids)
        pad_length = self.max_len - len(ids)
        ids = ids + [self.pad_token_id] * pad_length
        att = att + [0] * pad_length
        return torch.tensor(ids, dtype=torch.long), torch.tensor(att, dtype=torch.long)
