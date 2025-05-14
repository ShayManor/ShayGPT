import itertools
import math
import os
import random
import re
import shutil
import sys
import time
from select import select
from typing import Optional

import torch, torch.nn as nn
from datasets import load_dataset, DownloadConfig, interleave_datasets, Features, Value, load_from_disk
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.distributed as dist
from TextDataset import TextDataset, StreamDataset

from tokenizer import tokenizer, BOS_ID, EOS_ID, PAD_ID
from GPT import GPTConfig, GPT
import bitsandbytes as bnb
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume',
                   type=str,
                   default=None,
                   help='Path to a .pth checkpoint to load')
    p.add_argument('--epochs',
                   type=int,
                   default=50)
    p.add_argument('--batch_size',
                   type=int,
                   default=2)
    p.add_argument('--lr',
                   type=float,
                   default=5e-5)
    p.add_argument('--start',
                   type=int,
                   default=0,
                   )
    return p.parse_args()


def save(model, step):
    torch.save(
        model.module.state_dict(),
        f"checkpoint_step{step}.pth"
    )
    print(f"⚡ Saved checkpoint at step {step}")


def build_streams():
    token = os.getenv("HF_TOKEN")
    dl_cfg = DownloadConfig(max_retries=100, resume_download=True)
    feats = Features({"text": Value("string")})

    book = load_dataset("SamuelYang/bookcorpus", split="train",
                        streaming=True, download_config=dl_cfg, use_auth_token=token
                        ).select_columns(["text"]).cast(feats)

    mini = load_dataset("JeanKaddour/minipile", split="train",
                        streaming=True, download_config=dl_cfg, use_auth_token=token
                        ).select_columns(["text"]).cast(feats)

    gpt = load_dataset("terrycraddock/GPT2-PretrainV1-en", split="train",
                       streaming=True, download_config=dl_cfg, use_auth_token=token
                       ).select_columns(["text"]).cast(feats)

    wiki = load_dataset("google/wiki40b", "en", split="train",
                        streaming=True, download_config=dl_cfg, trust_remote_code=True, token=token
                        ).remove_columns(["wikidata_id", "version_id", "id"]).cast(feats)

    owt = load_dataset("Skylion007/openwebtext", split="train",
                       streaming=True, download_config=dl_cfg, trust_remote_code=True, token=token
                       ).cast(feats)

    oscar = load_dataset("oscar", "unshuffled_deduplicated_en", split="train",
                         streaming=True, download_config=dl_cfg, trust_remote_code=True, token=token
                         ).cast(Features({"id": Value("int64"), "text": Value("string")}))

    return {
        "oscar": oscar,
        "owt": owt,
        "wiki": wiki,
        "gpt": gpt,
        "mini": mini,
        "book": book,
    }


def train(resume: Optional[str],
          epochs: int = 3,
          batch_size: int = 16,
          lr: float = 5e-5,
          start: int = 0,
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

    # if PAD_ID is None:
    #     tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    #     PAD_ID = tokenizer.pad_token_id
    steps_per_epoch = 100
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, pad_id=PAD_ID)
    t0 = time.time()
    model = GPT(cfg)
    print("model built in", time.time() - t0, "seconds")
    if resume and os.path.isfile(resume):
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"⚡ Loaded weights from {resume}")
        del state
    t1 = time.time()
    model.to(device, non_blocking=True)
    print("moved to GPU in", time.time() - t1, "seconds")
    scaler = torch.amp.GradScaler('cuda')
    t2 = time.time()
    model = DistributedDataParallel(model, device_ids=[local_rank])
    print("wrapped in DDP in", time.time() - t2, "seconds")
    t3 = time.time()
    opt = bnb.optim.AdamW8bit(model.parameters(),
                              lr=lr,
                              betas=(0.9, 0.995),
                              weight_decay=0.01,
                              eps=1e-7)
    print(f'Made optimizer in {time.time() - t3} seconds')
    accum_steps = 16
    total_steps = steps_per_epoch * epochs
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_opt_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(0.15 * total_opt_steps)
    t4 = time.time()
    scheduler = get_linear_schedule_with_warmup(opt,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_opt_steps,
                                                )
    print(f'Created scheduler in {time.time() - t4} seconds')
    if PAD_ID is None:
        raise RuntimeError("PAD token not found in tokenizer!")
    global_step = 0
    losses = []

    def clean_example(ex):
        txt = ex["text"] if isinstance(ex, dict) else ex
        if len(txt) < 200:
            return False
        if re.search(r"<\/?html>|http[s]?://|\{.+?\}", txt):
            return False
        ascii_chars = sum(1 for c in txt if ord(c) < 128)
        return ascii_chars / len(txt) >= 0.95

    def collate_batch(texts):
        texts = [t["text"] if isinstance(t, dict) else t for t in texts]
        enc = tokenizer(texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512)
        return enc.input_ids, enc.attention_mask

    idx = 1
    while f'logfile_{idx}.txt' in os.listdir('data'):
        idx += 1
        print(idx)
    log_file = f'data/logfile_{idx}.txt'
    print(f"Opened logfile: {log_file}")
    open(log_file, 'x')
    STREAMS = build_streams()
    print(f"Wrote logfile")

    SCHEDULE = [
        (0, "oscar"),
        (20, "owt"),
        (25, "wiki"),
        (30, "gpt"),
        (33, "mini"),
        (35, "book"),
    ]

    def select_stream(epoch):
        corpus_name = max(e for e, _ in SCHEDULE if e <= epoch)
        corpus = dict(SCHEDULE)[corpus_name]
        base = STREAMS[corpus]

        iterator = (
            base
            .shuffle(buffer_size=256, seed=epoch)
            .shard(num_shards=world_size, index=rank)
        )
        per_epoch = steps_per_epoch * batch_size
        iterator = itertools.islice(iterator, per_epoch)

        return DataLoader(
            iterator,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_batch
        )

    try:
        for epoch in range(start, epochs):
            loader = select_stream(epoch)
            start_time = time.time()
            for step, (ids, attn_mask) in enumerate(loader):
                cur_time = time.time()
                ids = ids.to(device, non_blocking=True)
                attn_mask = attn_mask.bool()
                pad_mask = (attn_mask == 0)[:, :-1]
                input = ids[:, :-1]
                target = ids[:, 1:]
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input, pad_mask)
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_target = target.reshape(-1)
                    loss = nn.functional.cross_entropy(
                        flat_logits.float().clamp_(-100, 100),
                        flat_target,
                        ignore_index=PAD_ID,
                        label_smoothing=0.01,
                    )
                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() == scale_before:
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)

                if global_step % 100 == 0 and local_rank == 0:
                    losses.insert(0, loss.item())
                    avg_loss = sum(losses) / len(losses)
                    log = f"epoch {epoch} step {global_step} loss {sum(losses) / len(losses):.4f} lr = {scheduler.get_last_lr()[0]:.5} time = {time.time() - cur_time}"
                    with open(log_file, 'a') as f:
                        f.write(log)
                    print(log)
                    if len(losses) > 5:
                        losses.pop(-1)
                global_step += 1
                if step + 1 >= steps_per_epoch:
                    print(f"Tokens/step = {batch_size * 512 * accum_steps}")
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
