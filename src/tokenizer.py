from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

from src.clean import full_clean

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


def tokenize(texts):
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    # Post-processing so encode([s]) → [CLS] … [SEP]
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")),
                        ("[SEP]", tokenizer.token_to_id("[SEP]"))]
    )
    tokenizer.save("bpe-tokenizer.json")


if __name__ == '__main__':
    texts = full_clean('data/AI_Human.csv')
    tokenize(texts)
