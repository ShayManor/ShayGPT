# src/tokenizer.py
import itertools
import os

import torch
from tokenizers import Tokenizer
from pathlib import Path
from datasets import load_dataset

_TOKENIZER_PATH = Path(__file__).with_suffix(".json")


def collate_batch(texts, bos_id, eos_id, pad_id, max_len=512):
    # encode_batch returns a list[Encoding]
    encodings = tokenizer.encode_batch(texts)
    ids_list = []
    for enc in encodings:
        ids = enc.ids[: max_len - 2]
        ids = [bos_id] + ids + [eos_id]
        ids_list.append(torch.tensor(ids, dtype=torch.long))

    longest = max(x.size(0) for x in ids_list)
    padded = [torch.cat([x, x.new_full((longest - x.size(0),), pad_id)])
              for x in ids_list]
    return torch.vstack(padded)


if _TOKENIZER_PATH.exists():
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
else:
    from tokenizers import models, trainers, pre_tokenizers, processors
    from clean import full_clean


    def pj_iter():
        # data_files = {"train": "redpajama-cc/*.jsonl.zst"}
        ds_stream = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", trust_remote_code=True, streaming=True)
        for rec in ds_stream:
            yield rec["text"]


    print("⏳  Training BPE tokenizer on RedPajama CC stream...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=50_000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    sample_iter = itertools.islice(pj_iter(), 10_000_000)
    # shards = [x for x in os.listdir('redpajama') if x.endswith('jsonl')]
    # texts = full_clean(sample_iter)
    tokenizer.train_from_iterator(sample_iter, trainer=trainer)

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
    BOS_ID = tokenizer.token_to_id("[BOS]")
    EOS_ID = tokenizer.token_to_id("[EOS]")
    PAD_ID = tokenizer.token_to_id("[PAD]")
