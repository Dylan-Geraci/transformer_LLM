#!/usr/bin/env python3
"""
transformer_classifier.py

Transformer encoder-only classifier using precomputed skip-gram article embeddings.

Updated fix: ensures model embedding dim is divisible by num_heads by optionally projecting
the input token_dim to an embed_dim that is a multiple of num_heads.
"""

import argparse
import ast
import csv
import math
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------
# Utils: data loader / parsing
# ---------------------------

COMMON_EMBED_DIMS = [300, 200, 100, 50, 40, 32, 25, 20, 16, 8, 4, 2]


def parse_vector_string(s: str) -> List[float]:
    """
    Safely parse the 'news_vector' string column into a Python list of floats.
    Accepts strings like "[-0.12, 0.34, ...]".
    """
    if s is None:
        return []
    s = s.strip()
    try:
        vec = ast.literal_eval(s)
        if isinstance(vec, (list, tuple)):
            return [float(x) for x in vec]
    except Exception:
        # fallback: attempt to clean and split
        s2 = s.strip('[]" ')
        parts = [p for p in s2.split(',') if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except Exception:
            return []
    return []


def infer_seq_and_token_dim(flat_len: int) -> Tuple[int, int]:
    """
    Heuristic to infer (seq_len, token_dim) from flattened vector length.
    Tries common token dims (COMMON_EMBED_DIMS). If none match, returns (1, flat_len).
    """
    for d in COMMON_EMBED_DIMS:
        if flat_len % d == 0:
            seq_len = flat_len // d
            if 1 <= seq_len <= 1024:
                return seq_len, d
    # try factors <= 512
    for d in range(512, 1, -1):
        if flat_len % d == 0:
            seq_len = flat_len // d
            if 1 <= seq_len <= 1024:
                return seq_len, d
    # fallback: single token carrying entire vector
    return 1, flat_len


class NewsDataset(Dataset):
    def __init__(self, csv_path: str, max_seq_len: int = None):
        """
        Loads CSV at csv_path and parses news_vector and impact_score.
        """
        self.sequences = []
        self.labels = []

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                vec_str = row.get("news_vector") or row.get("news vector") or row.get("newsvector")
                label_str = row.get("impact_score") or row.get("impact") or row.get("impact score")
                if vec_str is None or label_str is None:
                    continue
                flat = parse_vector_string(vec_str)
                if len(flat) == 0:
                    continue
                # infer seq_len, token_dim
                seq_len, token_dim = infer_seq_and_token_dim(len(flat))
                arr = np.array(flat, dtype=np.float32)
                try:
                    arr = arr.reshape(seq_len, token_dim)
                except Exception:
                    arr = arr.reshape(1, -1)
                    seq_len, token_dim = arr.shape

                if max_seq_len is not None:
                    if seq_len > max_seq_len:
                        arr = arr[:max_seq_len, :]
                        seq_len = max_seq_len

                try:
                    label = int(float(label_str.strip().strip('"').strip("'")))
                except Exception:
                    try:
                        label = int(label_str)
                    except Exception:
                        continue

                self.sequences.append(arr)
                self.labels.append(label)

        if len(self.sequences) == 0:
            raise RuntimeError("No valid data parsed from CSV. Check column names and content.")

        self.token_dim = self.sequences[0].shape[1]
        self.max_seq_len = max(seq.shape[0] for seq in self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """
    batch: list of (np.ndarray seq (seq_len, token_dim), label)
    Returns padded tensors:
      - inputs: (batch, max_seq_len, token_dim)
      - lengths: tensor original seq_lens
      - labels: tensor (batch,)
    """
    seqs, labels = zip(*batch)
    batch_size = len(seqs)
    token_dim = seqs[0].shape[1]
    seq_lens = [s.shape[0] for s in seqs]
    max_len = max(seq_lens)
    padded = np.zeros((batch_size, max_len, token_dim), dtype=np.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0], :] = s
    inputs = torch.from_numpy(padded)  # (batch, max_len, token_dim)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(seq_lens, dtype=torch.long)
    return inputs, lengths, labels


# ------------------------------------
# Transformer encoder model (from-scratch)
# ------------------------------------

class TransformerEncoderLayerCustom(nn.Module):
    """
    Single Transformer Encoder layer with:
    - pre-LN (LayerNorm before attention and feedforward)
    - Multihead self-attention (nn.MultiheadAttention)
    - Feedforward MLP with GELU
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, batch_first=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.batch_first = batch_first

        # MultiheadAttention requires embed_dim % num_heads == 0
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=self.batch_first)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # x: (batch, seq_len, embed_dim) if batch_first True
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + attn_out

        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out
        return x


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        token_dim: int,
        max_seq_len: int,
        num_classes: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        positional_embedding_type: str = "learned",
        device: torch.device = None,
    ):
        super().__init__()
        self.token_dim = token_dim  # original input token dim
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.device = device or torch.device("cpu")

        # Choose embed_dim that's divisible by num_heads
        embed_dim = token_dim
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim + num_heads - 1) // num_heads) * num_heads
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # input projection if necessary (token_dim -> embed_dim)
        if self.embed_dim != self.token_dim:
            self.input_proj = nn.Linear(self.token_dim, self.embed_dim)
        else:
            self.input_proj = nn.Identity()

        # CLS token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.cls_token = None

        pos_count = max_seq_len + (1 if self.use_cls_token else 0)
        self.pos_embedding = nn.Embedding(pos_count, self.embed_dim)

        # feedforward dim default derived from embed_dim
        if ff_dim is None:
            ff_dim = max(self.embed_dim * 4, 128)
        self.ff_dim = ff_dim

        # Encoder layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerCustom(self.embed_dim, self.num_heads, self.ff_dim, dropout, batch_first=True) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x, lengths):
        """
        x: (batch, seq_len, token_dim)
        lengths: (batch,) original sequence lengths
        """
        batch, seq_len, token_dim = x.shape
        # allow token_dim to match self.token_dim (input), then project
        if token_dim != self.token_dim:
            raise ValueError(f"Unexpected input token_dim {token_dim}, expected {self.token_dim}")

        # project to model embedding dim
        x = self.input_proj(x)  # (batch, seq_len, embed_dim)

        # CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch, -1, -1)  # (batch,1,embed_dim)
            x = torch.cat([cls_tokens, x], dim=1)
            seq_len = seq_len + 1
            lengths = lengths + 1

        # positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb

        # key_padding_mask: True for padded positions
        max_len = seq_len
        device = x.device
        idxs = torch.arange(max_len, device=device).unsqueeze(0).expand(batch, max_len)
        key_padding_mask = idxs >= lengths.unsqueeze(1)  # True where padding

        # encoder stack
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # classification readout
        if self.use_cls_token:
            cls_out = x[:, 0, :]  # (batch, embed_dim)
            logits = self.classifier(cls_out)
        else:
            mask = ~key_padding_mask  # True for valid tokens
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked_x = x * mask
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = masked_x.sum(dim=1) / denom
            logits = self.classifier(pooled)

        return logits


# -----------------------
# Training / Evaluation
# -----------------------

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for inputs, lengths, labels in dataloader:
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(logits.detach(), labels)
        bsize = labels.size(0)
        total_loss += loss.item() * bsize
        total_acc += acc * bsize
        n += bsize
    return total_loss / n, total_acc / n


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    with torch.no_grad():
        for inputs, lengths, labels in dataloader:
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            acc = compute_accuracy(logits, labels)
            bsize = labels.size(0)
            total_loss += loss.item() * bsize
            total_acc += acc * bsize
            n += bsize
    return total_loss / n, total_acc / n


# ---------------------------
# Main script + CLI args
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Transformer encoder classifier on skip-gram article embeddings")
    p.add_argument("--csv_path", type=str, default="./datasets/vectorized_news_skipgram_embeddings.csv", help="Path to CSV file")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    p.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    p.add_argument("--ff_factor", type=float, default=4.0, help="Feedforward dim factor (ff_dim = max(embed_dim*factor, 128))")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    p.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_path", type=str, default="transformer_classifier.pth", help="Path to save trained model")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = NewsDataset(args.csv_path)
    print(f"Parsed {len(dataset)} examples; token_dim={dataset.token_dim}; max_seq_len={dataset.max_seq_len}")

    # label classes
    labels = sorted(set(dataset.labels))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    remapped = [label_to_idx[l] for l in dataset.labels]
    dataset.labels = remapped
    num_classes = len(labels)
    print(f"Detected label classes: {labels} -> mapped to 0..{num_classes-1}")

    # train/val/test split (80/10/10)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_train = max(n - 2, 1)
        n_val = 1
        n_test = n - n_train - n_val
    lengths = [n_train, n_val, n_test]
    train_set, val_set, test_set = random_split(dataset, lengths)
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    token_dim = dataset.token_dim
    max_seq_len = dataset.max_seq_len

    # compute embed_dim so it is divisible by num_heads (same logic used in the model)
    embed_dim = token_dim
    if embed_dim % args.num_heads != 0:
        embed_dim = ((embed_dim + args.num_heads - 1) // args.num_heads) * args.num_heads
    ff_dim = max(int(embed_dim * args.ff_factor), 128)

    model = TransformerEncoderClassifier(
        token_dim=token_dim,
        max_seq_len=max_seq_len,
        num_classes=num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=ff_dim,
        dropout=args.dropout,
        use_cls_token=True,
        positional_embedding_type="learned",
        device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "label_map": labels,
            }

    if best_state is not None:
        torch.save(best_state, args.save_path)
        print(f"Saved best model (val_acc={best_val_acc:.4f}) to {args.save_path}")
    else:
        print("No model saved (no improvement during training).")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"Test set | loss={test_loss:.4f} acc={test_acc:.4f}")

    print("Label mapping (original -> index):")
    for orig_label, idx in label_to_idx.items():
        print(f"  {orig_label} -> {idx}")


if __name__ == "__main__":
    main()
