from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id


def collate_batch(texts, bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID, max_len=512):
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    input_ids = batch.input_ids
    return input_ids, batch.attention_mask
