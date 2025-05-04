import torch
from torch import nn

from src.tokenizer import tokenizer


@torch.no_grad()
def sample(model, prompt, max_new_tokens=50, temperature=1.0):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        idx = torch.tensor([ids], device=next(model.parameters()).device)
        logits = model(idx)[:, -1, :] / temperature
        next_id = torch.multinomial(nn.functional.softmax(logits, dim=-1), num_samples=1).item()
        ids.append(next_id)
        if next_id == tokenizer.token_to_id("[EOS]"):
            break
    return tokenizer.decode(ids)

if __name__ == '__main__':
    sample()