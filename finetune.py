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
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast

SEQ_LEN = 256
VOCAB_SIZE = 26_000

D_MODEL = 512
N_LAYERS = 10
N_HEADS = 8
D_FF = 4 * D_MODEL

DIFFUSION_STEPS = 128
DROPOUT = 0.1 # higher for finetuning

TRAIN_STEPS = 3_000
BATCH_SIZE = 32
GRAD_ACCUM = 1
LR = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

def format_instruction(instruction: str, input_text: str, output_text: str) -> str:
    user_msg = instruction
    if input_text:
        user_msg += f"\n{input_text}"
    return f"<|user|>\n{user_msg}\n<|assistant|>\n{output_text}\n<|end|>\n"

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
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
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

class InstructionDataset(Dataset):
    def __init__(self, json_file, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.examples = []
        for item in data:
            text = format_instruction(item.get("instruction", ""), item.get("input", ""), item.get("output", ""))
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
            self.examples.append(torch.tensor(ids, dtype=torch.long))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        return self.examples[idx]

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
        input_ids=input_ids, attention_mask=attention_mask, t=t, mask_token_id=MASK_ID, T=T,
        PAD_ID=PAD_ID, BOS_ID=BOS_ID, EOS_ID=EOS_ID
    )
    logits = model(noisy_ids, timesteps=t, attention_mask=attention_mask)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    pretrain_dir = "checkpoints/pretrain"
    if not os.path.exists(pretrain_dir):
        print(f"Error: Pretrained model not found at {pretrain_dir}. Please run pretrain.py first.")
        return

    tokenizer_dir = os.path.join(pretrain_dir, "tokenizer")
    hf_tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)

    PAD_ID  = hf_tokenizer.pad_token_id
    MASK_ID = hf_tokenizer.mask_token_id
    BOS_ID  = hf_tokenizer.bos_token_id
    EOS_ID  = hf_tokenizer.eos_token_id

    with open(os.path.join(pretrain_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)
    cfg_dict['dropout'] = DROPOUT
    cfg = DiffusionLMConfig(**cfg_dict)
    
    model = DiffusionTransformerLM(cfg)
    model.load_state_dict(torch.load(os.path.join(pretrain_dir, "model.pt"), weights_only=True))
    
    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)

    def collate_fn(batch):
        # Pad sequences to max length in batch
        max_len = max(len(ids) for ids in batch)
        padded = []
        for ids in batch:
            pad_len = max_len - len(ids)
            padded.append(F.pad(ids, (0, pad_len), value=PAD_ID))
        input_ids = torch.stack(padded, dim=0)
        attention_mask = (input_ids != PAD_ID)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    dataset = InstructionDataset("instruction-data.json", hf_tokenizer, SEQ_LEN)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = 10 if args.smoke_test else TRAIN_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1 if args.smoke_test else WARMUP_STEPS, num_training_steps=total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    def eval_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                loss = diffusion_loss(model, batch, cfg.diffusion_steps, MASK_ID, PAD_ID, BOS_ID, EOS_ID)
                gathered = accelerator.gather(loss.detach().float().reshape(1))
                losses.append(gathered.cpu())
        model.train()
        if len(losses) == 0: return float("nan")
        return torch.cat(losses).mean().item()

    model.train()
    pbar = tqdm(range(total_steps), disable=not accelerator.is_main_process)
    running = []
    best_val_loss = float('inf')
    patience_counter = 0

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
        if (step + 1) % 10 == 0 and accelerator.is_main_process:
            pbar.set_description(f"loss={np.mean(running[-10:]):.4f}")

        if (step + 1) % 200 == 0 and accelerator.is_main_process:
            val_l = eval_loss()
            print(f"\nStep {step+1} | val_loss ~ {val_l:.4f}")
            
            if val_l < best_val_loss:
                best_val_loss = val_l
                patience_counter = 0
                BEST_DIR = "checkpoints/finetune_best"
                os.makedirs(BEST_DIR, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(BEST_DIR, "model.pt"))
                with open(os.path.join(BEST_DIR, "config.json"), "w") as f: json.dump(cfg.__dict__, f, indent=2)
                hf_tokenizer.save_pretrained(os.path.join(BEST_DIR, "tokenizer"))
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print("Early stopping triggered.")
                    break

    if accelerator.is_main_process:
        OUT_DIR = "checkpoints/finetune"
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(OUT_DIR, "model.pt"))
        with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)
        hf_tokenizer.save_pretrained(os.path.join(OUT_DIR, "tokenizer"))
        print("Saved final finetune checkpoint to:", OUT_DIR)

if __name__ == "__main__":
    main()
