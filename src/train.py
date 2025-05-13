import itertools
import math
import os
import re
import sys
import time
from typing import Optional

import torch, torch.nn as nn
from datasets import load_dataset, DownloadConfig, interleave_datasets, Features, Value
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
                   default=2e-4)
    return p.parse_args()


def save(model, step):
    torch.save(
        model.module.state_dict(),
        f"checkpoint_step{step}.pth"
    )
    print(f"⚡ Saved checkpoint at step {step}")


def train(resume: Optional[str],
          epochs: int = 3,
          batch_size: int = 16,
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
    # if PAD_ID is None:
    #     tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    #     PAD_ID = tokenizer.pad_token_id
    steps_per_epoch = 10_000
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size, pad_id=PAD_ID)
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
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)
    total_opt_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = int(0.15 * total_opt_steps)
    scheduler = get_linear_schedule_with_warmup(opt,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_opt_steps,
                                                )
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
        t = tokenizer(texts,
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=512)
        return t.input_ids, t.attention_mask

    dl_cfg = DownloadConfig(max_retries=100, resume_download=True)
    token = os.getenv("HF_TOKEN")
    wiki_ds = load_dataset(
        "google/wiki40b",  # clean Wikipedia
        "en",
        split="train",
        download_config=dl_cfg,
        token=token,
        trust_remote_code=True,
    )
    wiki_ds = wiki_ds.rename_column("wikidata_id", "id")
    wiki_ds = wiki_ds.remove_columns(["version_id"])
    wiki_ds = wiki_ds.remove_columns(["id"])
    wiki_ds = wiki_ds.cast(Features({"text": Value("string")}))
    owt_ds = load_dataset(
        "Skylion007/openwebtext",  # OpenWebText replication
        split="train",
        download_config=dl_cfg,
        token=token,
        trust_remote_code=True,
    )
    uniform_feats = Features({"text": Value("string")})
    owt_ds = owt_ds.cast(uniform_feats)
    hf_stream = interleave_datasets(
        [wiki_ds, owt_ds],
        probabilities=[0.2, 0.8],
        stopping_strategy="all_exhausted"
    )

    hf_stream = hf_stream.shuffle()
    dataset = StreamDataset(hf_stream, world_size, rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=lambda batch: collate_batch(batch),
    )
    try:
        for epoch in range(epochs):
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
                    print(
                        f"epoch {epoch} step {global_step} loss {sum(losses) / len(losses):.4f} lr = {scheduler.get_last_lr()[0]:.5} time = {time.time() - cur_time}")
                    if len(losses) > 5:
                        losses.pop(-1)
                    if avg_loss < 1.2:
                        save(model, global_step)
                        return
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
