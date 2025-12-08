#!/usr/bin/env python3
"""
rag_tune.py

RAG (Retrieval-Augmented Generation) system using pretrained TinyLlama.
Uses semantic embeddings for retrieval instead of TF-IDF.
No fine-tuning - pure RAG approach.
"""

import argparse
import json
import os
import sys
import warnings
from typing import List, Tuple

# Third-party imports with dependency checks
try:
    import torch
except Exception as e:
    raise RuntimeError("This script requires PyTorch. Install with `pip install torch`.") from e

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    raise RuntimeError("This script requires transformers. Install with `pip install transformers`.") from e

try:
    import PyPDF2
except Exception as e:
    raise RuntimeError("This script requires PyPDF2. Install with `pip install PyPDF2`.") from e

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("This script requires numpy. Install with `pip install numpy`.") from e

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available: will use numpy-based cosine similarity.")


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
# Semantic embedding generation
# -------------------------


def encode_chunks_with_model(
    chunks: List[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Encode text chunks into embeddings using TinyLlama.

    Strategy:
    - Tokenize each chunk
    - Pass through model to get hidden states
    - Mean-pool over sequence dimension (attention-mask aware)
    - Return normalized embeddings for cosine similarity

    Returns:
        np.ndarray: Normalized embeddings of shape [num_chunks, hidden_dim]
    """
    model.eval()
    embeddings = []

    if verbose:
        print(f"Encoding {len(chunks)} chunks into embeddings (batch_size={batch_size})...")

    with torch.no_grad():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # Tokenize (use shorter max_length for embedding efficiency)
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,  # Shorter for embedding generation
                return_tensors="pt"
            ).to(device)

            # Get last hidden state
            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

            # Mean pooling (attention mask aware)
            mask = encoded['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
            masked_hidden = hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # [batch, hidden_dim]
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # [batch, 1]
            mean_pooled = sum_hidden / sum_mask

            embeddings.append(mean_pooled.cpu().numpy())

            if verbose and (i + batch_size) % 20 == 0:
                print(f"  Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    # Concatenate and normalize
    all_embeddings = np.vstack(embeddings)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    normalized = all_embeddings / norms.clip(min=1e-9)

    if verbose:
        print(f"Generated embeddings shape: {normalized.shape}")

    return normalized


# -------------------------
# Semantic retrieval
# -------------------------


def retrieve_top_k_semantic(
    query: str,
    model,
    tokenizer,
    chunk_embeddings: np.ndarray,
    raw_chunks: List[str],
    top_k: int,
    device: torch.device
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Retrieve top-k most similar chunks to query using semantic embeddings.

    Args:
        query: The question/query text
        model: The language model for encoding
        tokenizer: Tokenizer for the model
        chunk_embeddings: Pre-computed normalized chunk embeddings [num_chunks, hidden_dim]
        raw_chunks: List of chunk texts
        top_k: Number of chunks to retrieve
        device: torch device

    Returns:
        Tuple of (retrieved_chunks, top_indices, top_scores)
    """
    # Encode query (returns normalized embedding)
    query_embedding = encode_chunks_with_model(
        [query], model, tokenizer, device, batch_size=1, verbose=False
    )

    # Compute cosine similarity (embeddings are already normalized, so dot product = cosine)
    similarities = np.dot(chunk_embeddings, query_embedding.T).squeeze()

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    top_chunks = [raw_chunks[i] for i in top_indices]

    return top_chunks, top_indices, top_scores


# -------------------------
# Answer generation
# -------------------------


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate answer using pretrained model (no fine-tuning).
    Returns only the generated text (no prompt echo).

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Full prompt with context
        device: torch device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        repetition_penalty: Repetition penalty

    Returns:
        str: Generated answer (without prompt)
    """
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048  # Allow longer prompts with context
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=(temperature > 0.0),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Extract only new tokens (no prompt echo)
    gen_seq = outputs[0]
    if gen_seq.shape[0] > input_len:
        new_tokens = gen_seq[input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        # Fallback: try to split prompt from output
        full_text = tokenizer.decode(gen_seq, skip_special_tokens=True)
        prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        if prompt_text in full_text:
            answer = full_text.split(prompt_text)[-1].strip()
        else:
            answer = full_text.strip()

    if not answer:
        answer = "No answer was generated."

    return answer


# -------------------------
# Main pipeline
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="RAG system using pretrained TinyLlama with semantic retrieval"
    )

    # Model and data
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="HuggingFace model name")
    parser.add_argument("--pdf_path", type=str, default="./datasets/cpsc-handbook-2022.pdf",
                       help="Path to PDF file")
    parser.add_argument("--output_dir", type=str, default="./rag_out",
                       help="Output directory for results")

    # Chunking parameters
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Max tokens per chunk")
    parser.add_argument("--stride", type=int, default=128,
                       help="Chunk overlap in tokens")

    # Retrieval parameters
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of chunks to retrieve")
    parser.add_argument("--embedding_batch_size", type=int, default=4,
                       help="Batch size for embedding generation")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Sampling temperature")
    parser.add_argument("--gen_top_k", type=int, default=50,
                       help="Top-k for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p (nucleus) for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                       help="Repetition penalty")

    # Questions
    parser.add_argument("--questions", type=str, nargs="*", default=None,
                       help="Custom questions (optional)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.pdf_path):
        print(f"ERROR: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    # Device setup
    device = get_device()
    print(f"Device selected: {device}")
    print()

    # Load model and tokenizer (NO FINE-TUNING)
    print(f"Loading pretrained model: {args.model_name}")
    print("NOTE: Using pretrained model WITHOUT fine-tuning (pure RAG approach)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        print("Using fp16 precision on GPU")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully")
    print()

    # Load and chunk PDF
    print(f"Reading PDF: {args.pdf_path}")
    full_text = read_pdf_text_concat(args.pdf_path)
    if not full_text.strip():
        print("ERROR: No text extracted from PDF")
        sys.exit(1)
    print(f"Extracted {len(full_text):,} characters from PDF")
    print()

    # Create chunks
    print("Creating text chunks with sliding window...")
    input_ids_chunks, raw_chunks = build_token_chunks_explicit(
        full_text,
        tokenizer,
        max_length=args.max_length,
        stride=args.stride,
        verbose=True
    )

    if len(raw_chunks) == 0:
        print("ERROR: No chunks created")
        sys.exit(1)

    print(f"Successfully created {len(raw_chunks)} chunks")
    print()

    # Generate embeddings for all chunks
    print("="*60)
    print("GENERATING SEMANTIC EMBEDDINGS")
    print("="*60)
    chunk_embeddings = encode_chunks_with_model(
        raw_chunks,
        model,
        tokenizer,
        device,
        batch_size=args.embedding_batch_size,
        verbose=True
    )
    print("Embedding generation complete!")
    print()

    # Default questions (same as params_finetune.py)
    if not args.questions:
        questions = [
            "What are the core courses required for a computer science undergraduate degree?",
            "Describe the rules for completing a senior project, including prerequisites.",
            "What are the degree requirements for graduation?",
        ]
    else:
        questions = args.questions

    print("="*60)
    print(f"ANSWERING {len(questions)} QUESTIONS")
    print("="*60)
    print()

    # Generate answers
    results = {}
    for q_idx, q in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {q_idx}/{len(questions)}:")
        print(f"{q}")
        print(f"{'='*60}\n")

        # Retrieve relevant chunks
        print("Retrieving relevant chunks...")
        retrieved_chunks, indices, scores = retrieve_top_k_semantic(
            q, model, tokenizer, chunk_embeddings, raw_chunks, args.top_k, device
        )

        print(f"Retrieved chunk indices: {list(indices)}")
        print(f"Similarity scores: {[f'{s:.4f}' for s in scores]}")
        print()

        # Build context
        context = "\n\n".join(retrieved_chunks)

        # Build prompt (same template as params_finetune.py)
        prompt = (
            "You are an assistant specialized in answering questions using the provided excerpts from a university CS handbook.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{q}\n\n"
            "INSTRUCTIONS: Answer succinctly and refer to the handbook when appropriate. If the answer is not in the context, say 'Not found in context.'\n\n"
            "ANSWER:"
        )

        # Generate answer
        print("Generating answer...")
        answer = generate_answer(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=256,
            temperature=args.temperature,
            top_k=args.gen_top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )

        print(f"\nGenerated answer:")
        print(f"{answer}")
        print()

        # Store result
        results[q] = {
            "prompt": prompt,
            "answer": answer,
            "retrieved_context": context,
            "retrieval_method": "semantic_embeddings",
            "chunk_indices": [int(i) for i in indices],
            "similarity_scores": [float(s) for s in scores]
        }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "rag_answers.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Output file: {out_json}")
    print()
    print("RAG system execution complete!")
    print()
    print("Next steps:")
    print("1. Compare with fine-tuned results: ft_out/finetune_answers.json")
    print("2. Analyze retrieval quality and answer quality")
    print("3. Use for assignment report (Part 4 evaluation)")


if __name__ == "__main__":
    main()
