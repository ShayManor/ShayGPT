# src/tokenizer.py
from tokenizers import Tokenizer
from pathlib import Path

_TOKENIZER_PATH = Path(__file__).with_suffix(".json")

if _TOKENIZER_PATH.exists():
    tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
else:
    from tokenizers import models, trainers, pre_tokenizers, processors
    from clean import full_clean

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=32_000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]"],
    )
    texts = full_clean("data/AI_Human.csv")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    pad_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(pad_token="[PAD]", pad_id=pad_id, direction="right")
    tokenizer.enable_truncation(max_length=128)

    tokenizer.save(str(_TOKENIZER_PATH))
