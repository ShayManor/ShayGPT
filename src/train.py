import os
import torch, torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
import torch.distributed as dist
from TextDataset import TextDataset
from GPT import GPTConfig, GPT
from tokenizer import tokenizer, collate_batch, BOS_ID, EOS_ID, PAD_ID
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup


def train(csv_path,
          epochs: int = 3,
          batch_size: int = 32,
          lr: float = 1.5e-4):
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print("Using device:", device)

    dataset = TextDataset(csv_path, max_len=512)
    bos_id, eos_id, pad_id = (tokenizer.token_to_id(t) for t in ["[BOS]", "[EOS]", "[PAD]"])
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        sampler=sampler,
                        num_workers=4,
                        collate_fn=lambda ex: collate_batch(ex, BOS_ID, EOS_ID, PAD_ID))

    cfg = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPT(cfg).to(device)
    model = torch.compile(model, mode="max-autotune")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = bnb.optim.AdamW8bit(model.parameters(),
                              lr=lr,
                              betas=(0.9, 0.95),
                              weight_decay=0.1,
                              eps=1e-8)
    total_steps = len(loader) * epochs
    warmup_steps = int(0.02 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    # scaler = GradScaler(init_scale=2 ** 8, growth_interval=50)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("PAD token not found in tokenizer!")
    global_step = 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for step, (ids, att) in enumerate(loader):
            ids = ids.to(device, non_blocking=True)
            # att = att.to(device, non_blocking=True)
            input = ids[:, :-1]
            target = ids[:, 1:]
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input)
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_target = target.reshape(-1)

                loss = nn.functional.cross_entropy(
                    flat_logits,
                    flat_target,
                    ignore_index=pad_id,
                    label_smoothing=0.0,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)

            if global_step % 1000 == 0 and local_rank == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f} lr = {scheduler.get_last_lr()[0]:.2e}")
            global_step += 1
        if local_rank == 0:
            torch.save(model.state_dict(), f"gpt_epoch{epoch}.pth")

    torch.save(model.module.tok_emb.weight.cpu(), "embed_matrix.pth")
    print("⚡ Training done.  Final loss:", loss.item())


if __name__ == "__main__":
    shards = [x for x in os.listdir('redpajama') if x.endswith('jsonl')]
    train(shards, epochs=10)


