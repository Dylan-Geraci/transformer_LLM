#!/usr/bin/env python3
"""
extractive_summary.py

Extractive summarization by sentence similarity using Skip-gram embeddings (if available)
or fallbacks.

Fix: increased csv.field_size_limit to handle very large article fields.
"""

import argparse
import ast
import csv
import os
import re
import sys
from typing import List

import numpy as np

# Increase CSV field size limit to handle very long article fields
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # some platforms (Windows) raise OverflowError for sys.maxsize; fallback to a large number
    csv.field_size_limit(10**7)

# Optional imports
try:
    from gensim.models import KeyedVectors, Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except Exception:
    GENSIM_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def read_csv_articles(csv_path: str):
    articles = []
    headlines = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            art = row.get("article") or row.get("body") or ""
            headline = row.get("headline") or ""
            if art and art.strip():
                articles.append(art.strip())
                headlines.append(headline.strip())
    return articles, headlines


def sentence_split(text: str) -> List[str]:
    text = text.strip().replace("\n", " ")
    splits = re.split(r'(?<=[.!?])\s+', text)
    out = []
    for s in splits:
        s = s.strip()
        if not s:
            continue
        if len(s) > 400 and ";" in s:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            out.extend(parts)
        else:
            out.append(s)
    return out


def tokenize_for_gensim(sent: str):
    return simple_preprocess(sent) if GENSIM_AVAILABLE else re.findall(r"\b\w+\b", sent.lower())


def load_word_vectors_kv(path: str):
    if not GENSIM_AVAILABLE:
        raise RuntimeError("gensim not available to load embeddings.")
    try:
        kv = KeyedVectors.load(path, mmap='r')
        return kv
    except Exception:
        try:
            kv = KeyedVectors.load_word2vec_format(path, binary=True)
            return kv
        except Exception:
            kv = KeyedVectors.load_word2vec_format(path, binary=False)
            return kv


def train_skipgram_on_corpus(sentences: List[List[str]], vector_size: int = 100, epochs: int = 10):
    if not GENSIM_AVAILABLE:
        raise RuntimeError("gensim not available to train skip-gram.")
    print("Training small skip-gram Word2Vec model on corpus (fallback). This may take time...")
    model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=1, sg=1, workers=1, epochs=epochs)
    return model.wv


def sentence_embeddings_with_wordvecs(sentences: List[str], kv, use_tfidf=False):
    toks = [tokenize_for_gensim(s) for s in sentences]
    if use_tfidf and SKLEARN_AVAILABLE:
        tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf = tf.fit_transform(sentences)
        feature_names = tf.get_feature_names_out()
        word_to_idx = {w: i for i, w in enumerate(feature_names)}
    else:
        tfidf = None
        word_to_idx = None

    emb_dim = kv.vector_size
    sent_embs = []
    for i, tokens in enumerate(toks):
        vecs = []
        weights = []
        for w in tokens:
            wk = w
            if wk in kv:
                vecs.append(kv[wk])
                if tfidf is not None:
                    idx = word_to_idx.get(wk)
                    wgt = 1.0
                    if idx is not None:
                        try:
                            wgt = float(tfidf[i, idx])
                        except Exception:
                            wgt = 1.0
                    weights.append(wgt)
                else:
                    weights.append(1.0)
        if len(vecs) == 0:
            sent_embs.append(np.zeros(emb_dim, dtype=float))
        else:
            vecs = np.stack(vecs, axis=0)
            weights = np.array(weights, dtype=float)
            weights = weights / (weights.sum() + 1e-12)
            sent_embs.append((weights[:, None] * vecs).sum(axis=0))
    return np.stack(sent_embs, axis=0)


def extractive_summary_from_article(article_text: str, top_k: int = 3, emb_path: str = None):
    sentences = sentence_split(article_text)
    if len(sentences) == 0:
        return "", "none", []

    sent_embs = None
    used_method = None
    if emb_path:
        if os.path.exists(emb_path):
            try:
                kv = load_word_vectors_kv(emb_path)
                sent_embs = sentence_embeddings_with_wordvecs(sentences, kv, use_tfidf=SKLEARN_AVAILABLE)
                used_method = f"pretrained_wordvectors({emb_path})"
            except Exception as e:
                print("Failed to load provided embeddings:", e)
                sent_embs = None

    if sent_embs is None and GENSIM_AVAILABLE:
        corpus_sents = [tokenize_for_gensim(s) for s in sentences]
        try:
            kv = train_skipgram_on_corpus(corpus_sents, vector_size=100, epochs=20)
            sent_embs = sentence_embeddings_with_wordvecs(sentences, kv, use_tfidf=False)
            used_method = "trained_local_skipgram"
        except Exception as e:
            print("Could not train skip-gram (gensim):", e)
            sent_embs = None

    if sent_embs is None and SKLEARN_AVAILABLE:
        print("Falling back to TF-IDF sentence vectors (sklearn).")
        tf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = tf.fit_transform(sentences)
        sent_embs = X.toarray()
        used_method = "tfidf"

    if sent_embs is None:
        print("Falling back to character n-gram hashing (very approximate).")
        sent_embs = []
        for s in sentences:
            h = np.zeros(128, dtype=float)
            for i, ch in enumerate(s.lower()):
                h[(i * ord(ch)) % 128] += 1.0
            sent_embs.append(h)
        sent_embs = np.stack(sent_embs, axis=0)
        used_method = "char_hash"

    def cosine_sim_matrix(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        Mn = M / norms
        return Mn.dot(Mn.T)

    sim = cosine_sim_matrix(sent_embs)
    centrality = sim.sum(axis=1)
    top_k = min(top_k, len(sentences))
    top_idx = np.argsort(-centrality)[:top_k]
    top_idx_set = set(top_idx.tolist())
    selected = [s for i, s in enumerate(sentences) if i in top_idx_set]
    summary = " ".join(selected)
    return summary, used_method, sentences


def main():
    p = argparse.ArgumentParser("Extractive summarization via sentence similarity")
    p.add_argument("--csv", type=str, default="./datasets/all_news.csv", help="Path to all_news.csv")
    p.add_argument("--index", type=int, default=0, help="Article index (0-based) to summarize")
    p.add_argument("--top_k", type=int, default=3, help="Number of sentences in extractive summary")
    p.add_argument("--emb_path", type=str, default=None, help="Optional path to precomputed word vectors (gensim KeyedVectors or word2vec format)")
    p.add_argument("--out", type=str, default="extractive_summary.txt", help="Output file")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print("CSV file not found:", args.csv)
        sys.exit(1)

    articles, headlines = read_csv_articles(args.csv)
    if args.index < 0 or args.index >= len(articles):
        print(f"Index {args.index} out of range (0..{len(articles)-1})")
        sys.exit(1)

    article = articles[args.index]
    headline = headlines[args.index] if args.index < len(headlines) else ""

    print(f"Article index {args.index}: headline (truncated): {headline[:200]!r}")
    summary, method, sentences = extractive_summary_from_article(article, top_k=args.top_k, emb_path=args.emb_path)

    print("Method used:", method)
    print("\n=== Extractive summary ===\n")
    print(summary)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("Headline:\n")
        f.write(headline + "\n\n")
        f.write("Extractive summary:\n")
        f.write(summary + "\n\n")
        f.write("Original sentences (for reference):\n")
        for i, s in enumerate(sentences):
            f.write(f"[{i}] {s}\n")
    print(f"\nSaved extractive summary to {args.out}")


if __name__ == "__main__":
    main()
