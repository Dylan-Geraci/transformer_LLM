#!/usr/bin/env python3
"""
params_finetune.py

Fine-tune TinyLlama (or similar HF model) on handbook with PEFT methods.

Key fixes in this version:
 - Robust chunking: explicit token-level sliding-window chunking (always produces overlapping windows
   from the full token sequence). This avoids the "only 1 chunk" problem and gives meaningful
   retrieval + fine-tuning signal for multi-page PDFs.
 - Adapter fallback (if AdapterConfig not present in peft).
 - Safe TrainingArguments builder for different transformers versions.
 - Generation returns only newly-generated tokens (no prompt echo).
"""

import argparse
import json
import math
import os
import sys
import warnings
from typing import List

# raise csv field limit if needed
try:
    import csv
    csv.field_size_limit(sys.maxsize)
except Exception:
    pass

# third-party imports
try:
    import torch
except Exception as e:
    raise RuntimeError("This script requires PyTorch. Install with `pip install torch`.") from e

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
except Exception as e:
    raise RuntimeError("This script requires transformers. Install with `pip install transformers`.") from e

# peft imports with adapter fallback
try:
    from peft import get_peft_model, LoraConfig, PrefixTuningConfig, TaskType
    try:
        from peft import AdapterConfig  # type: ignore
        ADAPTER_CONFIG_AVAILABLE = True
    except Exception:
        AdapterConfig = None  # type: ignore
        ADAPTER_CONFIG_AVAILABLE = False
except Exception as e:
    raise RuntimeError("This script requires peft. Install with `pip install peft`.") from e

try:
    import PyPDF2
except Exception as e:
    raise RuntimeError("This script requires PyPDF2. Install with `pip install PyPDF2`.") from e

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available: retrieval will be disabled (pip install scikit-learn).")


# -------------------------
# Helpers
# -------------------------


def get_device():
    """Try CUDA then fallback to CPU (small allocation test)."""
    if torch.cuda.is_available():
        try:
            torch.tensor([0], device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print("CUDA present but allocation failed; falling back to CPU:", e)
    return torch.device("cpu")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------
# PDF reading and robust chunking
# -------------------------


def read_pdf_text_concat(pdf_path: str) -> str:
    """Concatenate all non-empty page texts into one long string (keeps ordering)."""
    reader = PyPDF2.PdfReader(pdf_path)
    pages = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t and t.strip():
            pages.append(t.strip())
    return "\n\n".join(pages)


def build_token_chunks_explicit(
    full_text: str,
    tokenizer,
    max_length: int = 512,
    stride: int = 128,
    min_chunk_len: int = 20,
    verbose: bool = True,
):
    """
    Robust token-level sliding-window chunker.

    Steps:
      - Tokenize the full_text to a token id list (no truncation).
      - Create windows of size `max_length` with step `max_length - stride`.
      - Decode each token-window back to text for retrieval indexing.

    Returns (input_ids_chunks, raw_text_chunks)
    """
    if verbose:
        print(f"Tokenizing full text to ids (this may take a moment) ...")

    # Use tokenizer to obtain full id list; disable truncation, include no special tokens
    enc_all = tokenizer(
        full_text,
        truncation=False,
        return_overflowing_tokens=False,
        add_special_tokens=False,
    )
    full_ids = enc_all["input_ids"]
    total_tokens = len(full_ids)
    if verbose:
        print(f"Total tokens in document: {total_tokens:,}")

    if total_tokens == 0:
        return [], []

    window = max_length
    step = max(1, window - stride)
    ids_chunks = []
    raw_text_chunks = []

    # slide windows
    for start in range(0, total_tokens, step):
        end = start + window
        chunk_ids = full_ids[start:end]
        if len(chunk_ids) < min_chunk_len:
            continue
        ids_chunks.append(chunk_ids)
        # decode chunk back to string for retrieval index (skip special tokens)
        raw_text_chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True).strip())
        if end >= total_tokens:
            break

    if verbose:
        print(f"Created {len(ids_chunks)} token windows (window={window}, stride={stride}).")

    return ids_chunks, raw_text_chunks


# -------------------------
# Dataset wrapper
# -------------------------


class CausalLMDataset(torch.utils.data.Dataset):
    def __init__(self, inputs: List[List[int]]):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        ids = self.inputs[idx]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(ids, dtype=torch.long)}


# -------------------------
# PEFT creation with adapter fallback
# -------------------------


def create_peft_model(model, method: str, peft_kwargs: dict):
    method = method.lower()
    if method == "lora":
        lora_config = LoraConfig(
            r=peft_kwargs.get("r", 8),
            lora_alpha=peft_kwargs.get("alpha", 32),
            target_modules=peft_kwargs.get("target_modules", None),
            lora_dropout=peft_kwargs.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_config), "lora", lora_config

    if method in ("prefix", "prefix_tuning"):
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_kwargs.get("num_virtual_tokens", 30),
        )
        return get_peft_model(model, prefix_config), "prefix", prefix_config

    if method in ("adapter",):
        if ADAPTER_CONFIG_AVAILABLE:
            adapter_config = AdapterConfig(
                mh_adapter=True,
                output_adapter=True,
                reduction_factor=peft_kwargs.get("reduction_factor", 16),
                non_linearity=peft_kwargs.get("non_linearity", "relu"),
                task_type=TaskType.CAUSAL_LM,
            )
            return get_peft_model(model, adapter_config), "adapter", adapter_config
        else:
            warnings.warn(
                "AdapterConfig not found in installed peft. Falling back to a small LoRA configuration to emulate adapter behavior. "
                "If you want real Adapter support, upgrade peft (pip install -U peft)."
            )
            lora_config = LoraConfig(
                r=peft_kwargs.get("r", 4),
                lora_alpha=peft_kwargs.get("alpha", 16),
                target_modules=peft_kwargs.get("target_modules", None),
                lora_dropout=peft_kwargs.get("dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            return get_peft_model(model, lora_config), "adapter_emulated_with_lora", lora_config

    if method in ("full", "finetune", "full_finetune"):
        return model, "full", None

    raise ValueError(f"Unknown method: {method}")


# -------------------------
# Retrieval helpers
# -------------------------


def build_tfidf_retriever(raw_chunks: List[str]):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for retrieval. Install via `pip install scikit-learn`.")
    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    X = vectorizer.fit_transform(raw_chunks)
    return vectorizer, X


def retrieve_top_k_chunks(query: str, vectorizer, X, raw_chunks: List[str], top_k=3):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return [raw_chunks[i] for i in idxs], idxs, sims[idxs]


# -------------------------
# TrainingArguments helper (robust)
# -------------------------


def make_training_arguments(**kwargs):
    import inspect

    ctor = TrainingArguments.__init__
    sig = inspect.signature(ctor)
    allowed = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
    filtered = {}
    for k, v in kwargs.items():
        if k in allowed:
            filtered[k] = v
    return TrainingArguments(**filtered)


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama on handbook with PEFT methods")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--pdf_path", type=str, default="./datasets/cpsc-handbook-2022.pdf")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "prefix", "adapter", "full"])
    parser.add_argument("--output_dir", type=str, default="./ft_out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--questions", type=str, nargs="*", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print("PDF file not found:", args.pdf_path)
        sys.exit(1)

    device = get_device()
    print("Device selected:", device)

    print("Loading tokenizer and model (may download from HF)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use dtype parameter when possible
    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Create PEFT model wrapper
    peft_kwargs = {}
    peft_model, peft_name, peft_config = create_peft_model(model, args.method, peft_kwargs)

    trainable = count_trainable_parameters(peft_model)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%) using method {peft_name}")

    # Read entire PDF text and build robust token-level chunks
    full_text = read_pdf_text_concat(args.pdf_path)
    if not full_text.strip():
        print("No text extracted from PDF; exiting.")
        return

    input_ids_chunks, raw_chunks = build_token_chunks_explicit(
        full_text, tokenizer, max_length=args.max_length, stride=args.stride, verbose=True
    )

    n_chunks = len(input_ids_chunks)
    if n_chunks == 0:
        print("No chunks created; exiting.")
        return
    print(f"Prepared {n_chunks} training windows.")

    dataset = CausalLMDataset(input_ids_chunks)

    # split or use full dataset
    if n_chunks < 2:
        train_dataset = dataset
        val_dataset = None
        print("Tiny dataset detected: training on all windows (no eval split).")
    else:
        n_train = max(int(0.9 * n_chunks), 1)
        n_val = n_chunks - n_train
        if n_val < 1:
            n_val = 1
            n_train = n_chunks - 1
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        print(f"Split into train={len(train_dataset)} val={len(val_dataset)}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    ta_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "evaluation_strategy": "epoch" if n_chunks >= 2 else "no",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": (device.type == "cuda"),
        "remove_unused_columns": False,
        "report_to": "none",
    }
    training_args = make_training_arguments(**ta_kwargs)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if n_chunks >= 2 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    try:
        peft_model.to(device)
    except Exception as e:
        print("Warning: failed to move model to device directly:", e)

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Saved to {args.output_dir}")

    # Build retrieval index if available
    if SKLEARN_AVAILABLE:
        vectorizer, X = build_tfidf_retriever(raw_chunks)
    else:
        vectorizer, X = None, None

    # default questions
    if not args.questions:
        questions = [
            "What are the core courses required for a computer science undergraduate degree?",
            "Describe the rules for completing a senior project, including prerequisites.",
            "What are the degree requirements for graduation?",
        ]
    else:
        questions = args.questions

    results = {}
    for q in questions:
        print("\nQuestion:", q)
        if vectorizer is not None and len(raw_chunks) > 0:
            ctxs, idxs, scores = retrieve_top_k_chunks(q, vectorizer, X, raw_chunks, top_k=3)
            context = "\n\n".join(ctxs)
            print(f"Retrieved top chunk indices: {list(idxs)} scores: {list(scores)}")
        else:
            context = "\n\n".join(raw_chunks[:3] if len(raw_chunks) >= 3 else raw_chunks)

        # build instruction-style prompt that clearly asks model to answer from context
        prompt = (
            "You are an assistant specialized in answering questions using the provided excerpts from a university CS handbook.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{q}\n\n"
            "INSTRUCTIONS: Answer succinctly and refer to the handbook when appropriate. If the answer is not in the context, say 'Not found in context.'\n\n"
            "ANSWER:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)

        gen = peft_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=(args.temperature > 0.0),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # extract only generated tokens (avoid prompt echo)
        input_len = inputs["input_ids"].shape[1]
        gen_seq = gen[0]
        # if generation produced more tokens than input, decode only the new portion
        if gen_seq.shape[0] > input_len:
            new_tokens = gen_seq[input_len:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            out = tokenizer.decode(gen_seq, skip_special_tokens=True)
            prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
            if prompt_text and prompt_text in out:
                answer = out.split(prompt_text)[-1].strip()
            else:
                answer = out.strip()

        if not answer:
            answer = "No answer was generated."

        print("Generated (truncated):", answer[:800])
        results[q] = {"prompt": prompt, "answer": answer, "retrieved_context": context}

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "finetune_answers.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved generated answers to {out_json}")

    print("\n=== Summary ===")
    print(f"Model: {args.model_name}")
    print(f"Method: {args.method} | Trainable params: {count_trainable_parameters(peft_model):,} / {total:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
