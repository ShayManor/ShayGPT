import os
import re
import sys
import time
from typing import Optional

import torch, torch.nn as nn
from datasets import load_dataset, DownloadConfig
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist
from TextDataset import TextDataset, StreamDataset
from GPT import GPTConfig, GPT
from tokenizer import tokenizer, collate_batch, BOS_ID, EOS_ID, PAD_ID
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup
import argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume',
                   type=str,
                   default=None,
                   help='Path to a .pth checkpoint to load')
    p.add_argument('--epochs',
                   type=int,
                   default=20)
    p.add_argument('--batch_size',
                   type=int,
                   default=16)
    p.add_argument('--lr',
                   type=float,
                   default=5e-5)
    return p.parse_args()


def save(model, step):
    torch.save(
        model.module.state_dict(),
        f"checkpoint_step{step}.pth"
    )
    print(f"⚡ Saved checkpoint at step {step}")


def train(resume: Optional[str],
          epochs: int = 3,
          batch_size: int = 2,
          lr: float = 5e-5,
          ):
    dist.init_process_group("nccl")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print("Using device:", device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    steps_per_epoch = 10_000
    bos_id, eos_id, pad_id = (tokenizer.token_to_id(t) for t in ["[BOS]", "[EOS]", "[PAD]"])
    cfg = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPT(cfg)
    if resume and os.path.isfile(resume):
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"⚡ Loaded weights from {resume}")
        del state
    model.to(device)
    scaler = torch.amp.GradScaler('cuda')
    model = DistributedDataParallel(model, device_ids=[local_rank])
    opt = bnb.optim.AdamW8bit(model.parameters(),
                              lr=lr,
                              betas=(0.9, 0.995),
                              weight_decay=0.01,
                              eps=1e-7)
    accum_steps = 16
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.02 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    pad_id = tokenizer.token_to_id("[PAD]")
    if pad_id is None:
        raise RuntimeError("PAD token not found in tokenizer!")
    global_step = 0
    losses = []
    stream = load_dataset("oscar",
                          "unshuffled_deduplicated_en",
                          trust_remote_code=True,
                          streaming=True)
    raw_iter = stream.take(10)  # HuggingFace streaming: take first 10 examples
    for rec in raw_iter:
        print(type(rec), rec if isinstance(rec, str) else list(rec.keys()),
              "len:", len(rec) if isinstance(rec, str) else len(rec.get("text", "")))
    def clean_example(ex):
        txt = ex["text"] if isinstance(ex, dict) else ex
        if len(txt) < 200:
            return False
        ascii_chars = sum(1 for c in txt if ord(c) < 128)
        return ascii_chars / len(txt) >= 0.95

    stream = stream.filter(clean_example, batched=False)

    dataset = StreamDataset(stream, world_size, rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=lambda ex: collate_batch(ex, BOS_ID, EOS_ID, PAD_ID),
    )
    try:
        for epoch in range(epochs):
            start_time = time.time()
            for step, ids in enumerate(loader):
                cur_time = time.time()
                ids = ids.to(device, non_blocking=True)
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
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()
                    opt.zero_grad(set_to_none=True)

                if global_step % 100 == 0 and local_rank == 0:
                    losses.insert(0, loss.item())
                    avg_loss = sum(losses) / len(losses)
                    print(
                        f"epoch {epoch} step {global_step} loss {sum(losses) / len(losses):.4f} lr = {scheduler.get_last_lr()[0]:.5} time = {time.time() - cur_time}")
                    if len(losses) > 5:
                        losses.pop(-1)
                    if avg_loss < 1.2:
                        save(model, global_step)
                        return
                global_step += 1
                if step + 1 >= steps_per_epoch:
                    print(f'Epoch time: {time.time() - start_time}')
                    break
            if local_rank == 0:
                save(model, global_step)
    except KeyboardInterrupt:
        save(model, global_step)
    torch.save(model.module.tok_emb.weight.cpu(), "embed_matrix.pth")
    if "loss" in locals():
        print("⚡ Training done.  Final loss:", loss.item())
    else:
        print("No batches loaded")


if __name__ == "__main__":
    args = get_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume
    )
