import os
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from TextDataset import TextDataset
from TransformerEncoderModel import TransformerEncoderModel
from GPT import GPTConfig, GPT
from tokenizer import tokenizer


def train(csv_path,
          epochs: int = 3,
          init_batch: int = 32,
          lr: float = 1e-4):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TextDataset(csv_path, max_len=256)
    loader = DataLoader(dataset,
                        batch_size=init_batch,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        )
    cfg = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * epochs, eta_min=lr / 50)
    scaler = GradScaler(init_scale=2 ** 8, growth_interval=50)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("PAD token not found in tokenizer!")
    global_step = 0
    for epoch in range(epochs):
        for step, (_, idx) in enumerate(loader):
            idx = idx.to(device, non_blocking=True)
            input = idx[:, :-1]
            target = idx[:, 1:]
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=pad_id,
                    label_smoothing=0.0,
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

            if global_step % 100 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f} lr = {lr}")
            global_step += 1

        torch.save(model.state_dict(), f"transformer_encoder_e{epoch}.pth")

    torch.save(model.embedding.weight.cpu(), "embed_matrix.pth")
    print("⚡ Training done.  Final loss:", loss.item())


if __name__ == "__main__":
    shards = [
        "redpajama-1B/c4_sample.jsonl",
        "redpajama-1B/cc_2023-06_sample.jsonl",
        "redpajama-1B/cc_2020-05_sample.jsonl",
    ]
    train(shards, epochs=10)
