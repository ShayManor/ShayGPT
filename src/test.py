import torch
from tokenizer import tokenizer, BOS_ID
from GPT import GPTConfig, GPT

cfg = GPTConfig(vocab_size=50_000)
model = GPT(cfg)
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {total_params:,} ~= {total_params / 1e6:.2f} M parameters")
model.load_state_dict(torch.load("data/checkpoint4.pth", map_location=torch.device('cpu')))
model.eval()

prompt = "10 words you would use to describe itai are"
ids = tokenizer.encode(prompt).ids
ids.insert(0, BOS_ID)
for _ in range(25):
    inp = torch.tensor([ids], dtype=torch.long)
    logits = model(inp)[:, -1, :]
    sep_id = tokenizer.token_to_id("[SEP]")
    logits[0, sep_id] = -1e9

    next_id = torch.multinomial(
        torch.softmax(logits / 10, dim=-1), num_samples=1).item()
    ids.append(next_id)
    print(next_id)
    # if next_id == tokenizer.token_to_id("[EOS]"):
    #     break
print(tokenizer.decode(ids))
