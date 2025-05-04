# src/tokenizer.py
import os

import torch
from tokenizers import Tokenizer
from pathlib import Path

_TOKENIZER_PATH = Path(__file__).with_suffix(".json")


def collate_batch(examples, bos_id, eos_id, pad_id, max_len=512):
    # tokenize here to get variable lengths
    batch_ids = []
    for text in examples:
        ids = tokenizer.encode(text).ids
        ids = ids[:max_len - 2]
        ids = [bos_id] + ids + [eos_id]
        batch_ids.append(torch.tensor(ids, dtype=torch.long))

    longest = max(x.size(0) for x in batch_ids)
    padded = [torch.cat([x, x.new_full((longest - x.size(0),), pad_id)])
              for x in batch_ids]
    return torch.vstack(padded)


if _TOKENIZER_PATH.exists():
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
else:
    from tokenizers import models, trainers, pre_tokenizers, processors
    from clean import full_clean

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=32_000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    shards = [x for x in os.listdir('redpajama') if x.endswith('jsonl')]
    texts = full_clean(shards)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    pad_id = tokenizer.token_to_id("[PAD]")
    # tokenizer.enable_padding(pad_token="[PAD]", pad_id=pad_id, direction="right")
    # tokenizer.enable_truncation(max_length=512)

    tokenizer.save(str(_TOKENIZER_PATH))
