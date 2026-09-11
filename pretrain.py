import os
import math
import time
import json
import random
import argparse
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from dataclasses import dataclass
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
import tiktoken

SEQ_LEN = 256
VOCAB_SIZE = 50304 # GPT-2 vocab (50257) + special tokens, padded to multiple of 64

D_MODEL = 512
N_LAYERS = 10
N_HEADS = 8
D_FF = 4 * D_MODEL

DIFFUSION_STEPS = 128
DROPOUT = 0.05

TRAIN_STEPS = 30_000 # Adjusted for a ~4 hour run
BATCH_SIZE = 16
GRAD_ACCUM = 8
LR = 5e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2_000

def get_native_token_iterator(data_dir, seq_len):
    files = []
    for root, _, fnames in os.walk(data_dir):
        for fname in fnames:
            if fname.endswith(".npy") or fname.endswith(".bin"):
                files.append(os.path.join(root, fname))
    
    if not files:
        raise ValueError(f"No .npy or .bin files found in {data_dir}.")
    
    for file in files:
        if file.endswith(".npy"):
            mmap = np.load(file, mmap_mode='r')
        else:
            mmap = np.memmap(file, dtype=np.uint16, mode='r')
            
        for i in range(0, len(mmap) - seq_len, seq_len):
            block = mmap[i:i+seq_len].astype(np.int64)
            yield torch.tensor(block, dtype=torch.long)

def setup_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    })
    
    TOKENIZER_DIR = "tokenizer_pretrain"
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    return tokenizer

@dataclass
class DiffusionLMConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    diffusion_steps: int

class DiffusionTransformerLM(nn.Module):
    def __init__(self, cfg: DiffusionLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, timesteps, attention_mask=None):
        B, L = input_ids.shape
        if L > self.cfg.seq_len:
            raise ValueError(f"Sequence length {L} > cfg.seq_len {self.cfg.seq_len}")

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        t_emb = self.time_emb(timesteps).unsqueeze(1)
        x = x + t_emb
        x = self.drop(x)

        if attention_mask is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = ~attention_mask

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

class TokenBlockIterableDataset(IterableDataset):
    def __init__(self, data_dir, seq_len):
        self.data_dir = data_dir
        self.seq_len = seq_len

    def __iter__(self):
        # We loop infinitely over the dataset generator
        while True:
            for block in get_native_token_iterator(self.data_dir, self.seq_len):
                yield block

def mask_ratio_schedule(t, T: int):
    return t.float() / float(T)

@torch.no_grad()
def corrupt_with_mask(input_ids, attention_mask, t, mask_token_id: int, T: int, PAD_ID, BOS_ID, EOS_ID):
    B, L = input_ids.shape
    ratio = mask_ratio_schedule(t, T).unsqueeze(1)

    can_mask = attention_mask.clone()
    can_mask &= (input_ids != BOS_ID) & (input_ids != EOS_ID) & (input_ids != PAD_ID)

    rand = torch.rand((B, L), device=input_ids.device)
    mask_positions = (rand < ratio) & can_mask

    noisy = input_ids.clone()
    noisy[mask_positions] = mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask_positions] = input_ids[mask_positions]

    return noisy, labels, mask_positions

def diffusion_loss(model, batch, T, MASK_ID, PAD_ID, BOS_ID, EOS_ID):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    B = input_ids.size(0)
    t = torch.randint(1, T + 1, (B,), device=input_ids.device)

    noisy_ids, labels, _ = corrupt_with_mask(
        input_ids=input_ids,
        attention_mask=attention_mask,
        t=t,
        mask_token_id=MASK_ID,
        T=T,
        PAD_ID=PAD_ID,
        BOS_ID=BOS_ID,
        EOS_ID=EOS_ID
    )

    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 10-step test")
    parser.add_argument("--data-dir", type=str, default="fineweb dataset", help="Path to dataset")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/pretrain if it exists")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="Number of steps to train")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Please download dataset to {data_dir} and extract the files.")
        return

    if not os.path.exists("tokenizer_pretrain/tokenizer.json") and not os.path.exists("tokenizer_pretrain/tokenizer_config.json"):
        hf_tokenizer = setup_tokenizer()
    else:
        hf_tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer_pretrain")

    PAD_ID  = hf_tokenizer.pad_token_id
    MASK_ID = hf_tokenizer.mask_token_id
    BOS_ID  = hf_tokenizer.bos_token_id
    EOS_ID  = hf_tokenizer.eos_token_id

    if args.resume and os.path.exists("checkpoints/pretrain/config.json"):
        print("Resuming from checkpoints/pretrain...")
        with open("checkpoints/pretrain/config.json", "r") as f:
            cfg_dict = json.load(f)
        cfg = DiffusionLMConfig(**cfg_dict)
        model = DiffusionTransformerLM(cfg)
        model.load_state_dict(torch.load("checkpoints/pretrain/model.pt", weights_only=True))
    else:
        cfg = DiffusionLMConfig(
            vocab_size=len(hf_tokenizer),
            seq_len=SEQ_LEN,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF,
            dropout=DROPOUT,
            diffusion_steps=DIFFUSION_STEPS,
        )
        model = DiffusionTransformerLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    # if hasattr(torch, "compile"):
    #     print("Compiling model...")
    #     model = torch.compile(model)

    def collate_blocks(batch):
        input_ids = torch.stack(batch, dim=0)
        attention_mask = (input_ids != PAD_ID)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    train_ds = TokenBlockIterableDataset(data_dir, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
    
    val_loader = DataLoader(TokenBlockIterableDataset(data_dir, SEQ_LEN), batch_size=BATCH_SIZE, collate_fn=collate_blocks)

    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")
    device = accelerator.device

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = 10 if args.smoke_test else args.steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1 if args.smoke_test else WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    def eval_loss(n_batches=10):
        model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= n_batches:
                    break
                loss = diffusion_loss(model, batch, cfg.diffusion_steps, MASK_ID, PAD_ID, BOS_ID, EOS_ID)
                gathered = accelerator.gather(loss.detach().float().reshape(1))
                losses.append(gathered.cpu())
        model.train()
        if len(losses) == 0: return float("nan")
        return torch.cat(losses).mean().item()

    model.train()
    pbar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    running = []

    train_iter = iter(train_loader)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss = diffusion_loss(model, batch, cfg.diffusion_steps, MASK_ID, PAD_ID, BOS_ID, EOS_ID) / GRAD_ACCUM
        accelerator.backward(loss)

        if (step + 1) % GRAD_ACCUM == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        running.append(loss.item() * GRAD_ACCUM)

        if (step + 1) % 50 == 0 and accelerator.is_main_process:
            pbar.set_description(f"loss={np.mean(running[-50:]):.4f} lr={scheduler.get_last_lr()[0]:.2e}")

        if (step + 1) % 500 == 0 and accelerator.is_main_process:
            val_l = eval_loss(n_batches=10)
            print(f"\nStep {step+1} | val_loss ~ {val_l:.4f}")
            
            if (step + 1) % 5000 == 0:
                OUT_DIR = f"checkpoints/pretrain_step_{step+1}"
                os.makedirs(OUT_DIR, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(OUT_DIR, "model.pt"))

    if accelerator.is_main_process:
        OUT_DIR = "checkpoints/pretrain"
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(OUT_DIR, "model.pt"))
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)
        hf_tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
        print("Saved final pretrain checkpoint to:", OUT_DIR)

if __name__ == "__main__":
    main()
