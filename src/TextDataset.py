import numpy as np
import pandas as pd
import torch
from torch import float32
from torch.utils.data import Dataset

from src.clean import full_clean
from src.tokenizer import tokenize, tokenizer


class TextDataset(Dataset):
    def __init__(self, path, max_len=128):
        self.texts = full_clean(path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.texts[idx])
        ids = encoded.ids[:self.max_len]
        att = [1] * len(ids)
        pad_length = self.max_len - len(ids)
        ids = ids + [self.tokenizer.token_to_id("[PAD]")] * pad_length
        att = att + [0] * pad_length
        return torch.tensor(ids), torch.tensor(att)
