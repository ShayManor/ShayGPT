import torch
from tokenizer import tokenizer
from GPT import GPTConfig, GPT

cfg = GPTConfig(
    vocab_size=tokenizer.get_vocab_size())
model = GPT(cfg)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} ~= {total_params / 1e6:.2f} M parameters")
model.load_state_dict(torch.load("data/e9.pth", map_location=torch.device('cpu')))
model.eval()

prompt = "Oliver is a beacon of "
ids = tokenizer.encode(prompt).ids
for _ in range(25):
    inp = torch.tensor([ids])
    logits = model(inp)[:, -1, :]
    next_id = torch.multinomial(
        torch.softmax(logits / 0.8, dim=-1), num_samples=1).item()
    ids.append(next_id)
    if next_id == tokenizer.token_to_id("[EOS]"):
        break
print(tokenizer.decode(ids))
