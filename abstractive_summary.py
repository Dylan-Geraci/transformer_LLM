#!/usr/bin/env python3
"""
abstractive_summary.py

Abstractive summarization (article -> headline) using a small encoder-decoder Transformer.

Fixes:
 - Robust device selection: tries CUDA and falls back to CPU if allocation fails.
 - Fixed device mismatch in greedy generation (no .cpu() mid-loop).
 - Keeps previous csv.field_size_limit fix for large article fields.
"""

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Increase CSV field size limit to handle very long article fields
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(10**7)

# -------------------------
# Utilities: tokenization/vocab
# -------------------------

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def simple_tokenize(text: str):
    text = text.strip().lower()
    toks = re.findall(r"\w+|[^\s\w]", text, re.UNICODE)
    return toks


def build_vocab(sentences: List[List[str]], min_freq: int = 1, max_size: int = 20000):
    counter = Counter()
    for s in sentences:
        counter.update(s)
    items = [t for t, c in counter.items() if c >= min_freq]
    items.sort(key=lambda t: (-counter[t], t))
    items = items[: max(0, max_size - len(SPECIAL_TOKENS))]
    idx = dict(SPECIAL_TOKENS)
    cur = len(idx)
    for t in items:
        if t not in idx:
            idx[t] = cur
            cur += 1
    return idx


def numericalize(tokens: List[str], vocab: dict, max_len: int, add_sos_eos=True):
    ids = []
    if add_sos_eos:
        ids.append(vocab["<sos>"])
    for t in tokens[: max_len - (2 if add_sos_eos else 0)]:
        ids.append(vocab.get(t, vocab["<unk>"]))
    if add_sos_eos:
        ids.append(vocab["<eos>"])
    if len(ids) < max_len:
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]


# -------------------------
# Dataset
# -------------------------


class ArticleHeadlineDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], vocab_src: dict, vocab_tgt: dict, max_src_len: int = 256, max_tgt_len: int = 32):
        self.pairs = pairs
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_tokens = simple_tokenize(src)
        tgt_tokens = simple_tokenize(tgt)
        src_ids = numericalize(src_tokens, self.vocab_src, self.max_src_len, add_sos_eos=False)
        tgt_ids = numericalize(tgt_tokens, self.vocab_tgt, self.max_tgt_len, add_sos_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs = torch.stack(srcs, dim=0)
    tgts = torch.stack(tgts, dim=0)
    return srcs, tgts


# ---------------------------
# Model: small encoder-decoder using nn.Transformer
# ---------------------------


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), seq_len)
        return self.pos_emb(pos)


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_src_len: int = 256,
        max_tgt_len: int = 32,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos_emb = PositionalEmbedding(d_model, max_src_len)
        self.tgt_pos_emb = PositionalEmbedding(d_model, max_tgt_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

    def forward(self, src_ids, tgt_ids, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.src_tok_emb(src_ids) * math.sqrt(self.d_model)
        src_emb = src_emb + self.src_pos_emb(src_ids)
        tgt_emb = self.tgt_tok_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.tgt_pos_emb(tgt_ids)
        tgt_len = tgt_ids.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src_emb.device)
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.generator(output)
        return logits

    def encode(self, src_ids, src_key_padding_mask=None):
        src_emb = self.src_tok_emb(src_ids) * math.sqrt(self.d_model)
        src_emb = src_emb + self.src_pos_emb(src_ids)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode_step(self, tgt_ids, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.tgt_tok_emb(tgt_ids) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb + self.tgt_pos_emb(tgt_ids)
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(out)
        return logits


# ---------------------------
# Training / Generation helpers
# ---------------------------


def create_key_padding_mask(ids, pad_idx):
    return ids == pad_idx


def greedy_generate(model: TransformerSeq2Seq, src_ids: torch.Tensor, src_pad_idx: int, tgt_vocab: dict, max_tgt_len: int, device):
    """
    Greedily generate headline tokens. All tensors stay on `device` during generation.
    """
    model.eval()
    src_ids = src_ids.to(device)
    pad_mask = create_key_padding_mask(src_ids, src_pad_idx).to(device)
    memory = model.encode(src_ids, src_key_padding_mask=pad_mask)

    batch = src_ids.size(0)
    sos = tgt_vocab["<sos>"]
    eos = tgt_vocab["<eos>"]
    pad = tgt_vocab["<pad>"]

    ys = torch.full((batch, 1), sos, dtype=torch.long, device=device)
    for _ in range(max_tgt_len - 1):
        tgt_key_padding_mask = create_key_padding_mask(ys, pad).to(device)
        tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
        out_logits = model.decode_step(ys.to(device), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=pad_mask)
        next_logits = out_logits[:, -1, :]  # (batch, vocab)
        next_tokens = next_logits.argmax(dim=-1, keepdim=True)  # stays on device
        ys = torch.cat([ys, next_tokens], dim=1)  # both tensors on same device
        if (next_tokens == eos).all():
            break

    # convert to CPU numpy at the end
    ys_cpu = ys.cpu().numpy()
    inv_vocab = {v: k for k, v in tgt_vocab.items()}
    results = []
    for seq in ys_cpu:
        toks = []
        for idx in seq:
            if idx == sos or idx == pad:
                continue
            if idx == eos:
                break
            toks.append(inv_vocab.get(int(idx), "<unk>"))
        results.append(" ".join(toks))
    return results


def read_csv_pairs(csv_path: str, min_pairs: int = 50):
    pairs = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            art = row.get("article") or ""
            head = row.get("headline") or ""
            if art and head and art.strip() and head.strip():
                pairs.append((art.strip(), head.strip()))
    if len(pairs) < min_pairs:
        print(f"Warning: only found {len(pairs)} article/headline pairs (min_pairs requested {min_pairs}).")
    return pairs


def prepare_dataset(csv_path: str, num_pairs: int = 200, min_pairs: int = 50, max_src_len: int = 256, max_tgt_len: int = 32):
    pairs = read_csv_pairs(csv_path, min_pairs=min_pairs)
    if num_pairs < len(pairs):
        pairs = pairs[:num_pairs]
    src_tok = [simple_tokenize(a)[:max_src_len] for a, _ in pairs]
    tgt_tok = [simple_tokenize(h)[: max_tgt_len - 2] for _, h in pairs]
    vocab_src = build_vocab(src_tok, min_freq=1, max_size=40000)
    vocab_tgt = build_vocab(tgt_tok, min_freq=1, max_size=40000)
    dataset = ArticleHeadlineDataset(pairs, vocab_src, vocab_tgt, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
    return dataset, vocab_src, vocab_tgt


def train_model(model, train_loader, val_loader, device, epochs=5, lr=1e-3, pad_idx=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for src_ids, tgt_ids in train_loader:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            decoder_input = tgt_ids[:, :-1]
            target = tgt_ids[:, 1:]
            src_key_padding_mask = create_key_padding_mask(src_ids, pad_idx).to(device)
            tgt_key_padding_mask = create_key_padding_mask(decoder_input, pad_idx).to(device)
            logits = model(src_ids, decoder_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * src_ids.size(0)
            n += src_ids.size(0)
        train_loss = running_loss / max(1, n)
        model.eval()
        running_val = 0.0
        nval = 0
        with torch.no_grad():
            for src_ids, tgt_ids in val_loader:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)
                decoder_input = tgt_ids[:, :-1]
                target = tgt_ids[:, 1:]
                src_key_padding_mask = create_key_padding_mask(src_ids, pad_idx).to(device)
                tgt_key_padding_mask = create_key_padding_mask(decoder_input, pad_idx).to(device)
                logits = model(src_ids, decoder_input, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
                running_val += loss.item() * src_ids.size(0)
                nval += src_ids.size(0)
        val_loss = running_val / max(1, nval)
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {"model": model.state_dict()}
    return best_state, best_val_loss


# ---------------------------
# Device helper
# ---------------------------

def get_device():
    """
    Try to use CUDA if available and usable; otherwise fallback to CPU.
    We attempt a tiny CUDA allocation to ensure the device is actually allocatable.
    """
    if torch.cuda.is_available():
        try:
            # attempt a tiny allocation to verify CUDA is usable
            _ = torch.tensor([0], device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print("CUDA appears, but allocation failed — falling back to CPU. Error:", e)
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def main():
    p = argparse.ArgumentParser("Abstractive summarization via Transformer encoder-decoder")
    p.add_argument("--csv", type=str, default="./datasets/all_news.csv", help="Path to all_news.csv")
    p.add_argument("--num_pairs", type=int, default=200000, help="Number of article/headline pairs to use (>=50)")
    p.add_argument("--min_pairs", type=int, default=50, help="Minimum pairs to expect")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--d_model", type=int, default=128, help="Transformer d_model")
    p.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    p.add_argument("--num_enc_layers", type=int, default=2, help="Encoder layers")
    p.add_argument("--num_dec_layers", type=int, default=2, help="Decoder layers")
    p.add_argument("--max_src_len", type=int, default=256, help="Max tokens for article")
    p.add_argument("--max_tgt_len", type=int, default=32, help="Max tokens for headline (including sos/eos)")
    p.add_argument("--save_path", type=str, default="abstractive_transformer.pth", help="Model save path")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print("CSV not found:", args.csv)
        sys.exit(1)

    dataset, vocab_src, vocab_tgt = prepare_dataset(args.csv, num_pairs=args.num_pairs, min_pairs=args.min_pairs, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    n = len(dataset)
    if n == 0:
        print("No pairs found.")
        sys.exit(1)
    print(f"Using {n} pairs. Vocab sizes: src={len(vocab_src)} tgt={len(vocab_tgt)}")

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_train = max(n - 2, 1)
        n_val = 1
        n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = get_device()
    print("Device:", device)

    pad_idx = SPECIAL_TOKENS["<pad>"]
    model = TransformerSeq2Seq(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        dim_feedforward=max(512, args.d_model * 4),
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        pad_idx=pad_idx,
    )

    # attempt to move model to device, with fallback
    try:
        model.to(device)
    except Exception as e:
        print("Failed to move model to device:", e)
        print("Falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)

    best_state, best_val_loss = train_model(model, train_loader, val_loader, device, epochs=args.epochs, lr=1e-3, pad_idx=pad_idx)
    if best_state is not None:
        torch.save({"model_state": best_state["model"], "vocab_src": vocab_src, "vocab_tgt": vocab_tgt}, args.save_path)
        print(f"Saved model to {args.save_path} (val_loss={best_val_loss:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    # demo: generate for first article
    sample_src, sample_tgt = dataset.pairs[0]
    src_tokens = simple_tokenize(sample_src)[: args.max_src_len]
    src_ids = numericalize(src_tokens, vocab_src, args.max_src_len, add_sos_eos=False)
    src_tensor = torch.tensor([src_ids], dtype=torch.long)
    generated = greedy_generate(model, src_tensor, pad_idx, vocab_tgt, args.max_tgt_len, device=device)
    print("\n=== Example ===")
    print("Original headline:", sample_tgt)
    print("Generated headline:", generated[0])

    print("\nDone.")


if __name__ == "__main__":
    main()
