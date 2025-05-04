import os
import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from TextDataset import TextDataset
from TransformerEncoderModel import TransformerEncoderModel
from tokenizer import tokenizer


def train(csv_path: str,
          epochs: int = 3,
          init_batch: int = 64,
          lr: float = 2e-4):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TextDataset(csv_path, max_len=128)
    loader = DataLoader(dataset,
                        batch_size=init_batch,
                        shuffle=True,
                        num_workers=os.cpu_count() // 2,
                        pin_memory=True,
                        persistent_workers=True)

    model = TransformerEncoderModel(tokenizer.get_vocab_size()).to(device)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    scaler = GradScaler()

    global_step = 0
    for epoch in range(1, epochs + 1):
        for batch_ids, batch_att in loader:
            batch_ids = batch_ids.to(device, non_blocking=True)
            batch_att = batch_att.to(device, non_blocking=True)

            inp = batch_ids[:, :-1]
            tgt = batch_ids[:, 1:]
            att = batch_att[:, :-1]

            with autocast():
                logits = model(inp, att) @ model.embedding.weight.T
                loss = crit(logits.view(-1, logits.size(-1)),
                            tgt.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            if global_step % 100 == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
            global_step += 1

        torch.save(model.state_dict(),
                   f"transformer_encoder_e{epoch}.pth")

    torch.save(model.embedding.weight.cpu(), "embed_matrix.pth")
    print("⚡ Training done.  Final loss:", loss.item())



if __name__ == "__main__":
    train("data/AI_Human.csv", epochs=6)
