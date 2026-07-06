"""
build_faiss.py — Stage 2b: build the retrieval indexes.

Consumes the artifacts from embed.py and produces:

    faiss.index      dense index (exact cosine via inner product on unit vectors)
    bm25.pkl         lexical index over the same chunks (for hybrid retrieval)

Both are keyed to chunks.jsonl by row order, so a hit (dense or lexical) maps
straight back to its paragraph. At ~1k-5k paragraphs an exact FlatIP index is
the right call: it's brute-force but sub-millisecond at this scale and avoids
the recall loss of an approximate (IVF/HNSW) index. If the corpus later grows
past ~100k chunks, switch to IndexIVFFlat (see note in build_dense_index).

This whole step is CPU-friendly (faiss-cpu) — it does not need the GPU, which
keeps the "retrieval runs on the laptop" story intact.

Usage:
    python -m src.index.build_faiss
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.io import load_config, read_jsonl, resolve


# Tokenizer for BM25. Keeps hyphenated/period-joined doctrine tokens intact
# (e.g. "METT-TC", "FM 5-0", "ATP 5-0.1") instead of shattering them, since
# acronym matching is exactly where lexical retrieval earns its keep here.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-.]*[A-Za-z0-9]|[A-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_dense_index(embeddings: np.ndarray) -> faiss.Index:
    """Exact inner-product index over unit-normalized vectors (== cosine)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # --- to scale past ~100k chunks, swap the two lines above for: -------------
    #   quantizer = faiss.IndexFlatIP(dim)
    #   index = faiss.IndexIVFFlat(quantizer, dim, nlist=256,
    #                              faiss.METRIC_INNER_PRODUCT)
    #   index.train(embeddings)
    # --------------------------------------------------------------------------
    index.add(embeddings)
    return index


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def build_indexes(config: dict) -> None:
    index_dir = resolve(config["paths"]["index"])

    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"
    if not emb_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path.name}/{chunks_path.name} in {index_dir}. "
            "Run `python -m src.index.embed` first."
        )

    embeddings = np.load(emb_path)
    chunks = read_jsonl(chunks_path)
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: {embeddings.shape[0]} embeddings vs "
            f"{len(chunks)} chunks. Re-run embed.py to regenerate both."
        )

    # Dense
    dense = build_dense_index(embeddings)
    faiss.write_index(dense, str(index_dir / "faiss.index"))
    print(f"Dense FAISS index: {dense.ntotal} vectors, dim {embeddings.shape[1]}")

    # Lexical
    bm25 = build_bm25_index(chunks)
    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 index: {len(chunks)} documents")

    print(f"Indexes written to {index_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()
    build_indexes(load_config(args.config))


if __name__ == "__main__":
    main()