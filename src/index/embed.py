"""
embed.py — Stage 2a: encode parsed doctrine paragraphs into dense vectors.

Runs on the A100 (Colab): batch-embeds every paragraph in the corpus with a
sentence-transformers model, L2-normalizes (so inner product == cosine), and
writes two aligned artifacts to data/index/:

    embeddings.npy   float32 [N, D], row i corresponds to chunks.jsonl line i
    chunks.jsonl     the chunk store, same order as the embedding rows

Keeping the embeddings and the chunk store row-aligned is what lets the FAISS
index map a hit back to its paragraph id/text later.

Usage:
    python -m src.index.embed                     # uses config.yaml
    python -m src.index.embed --batch-size 128
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import load_config, load_corpus, resolve, write_jsonl


def build_embedding_text(chunk: dict) -> str:
    """
    Text that actually gets embedded for a paragraph.

    Baseline = raw paragraph text. We prepend the section header when present,
    because doctrine paragraphs are often terse and the section title adds
    disambiguating context (e.g. many paragraphs say "The commander..." — the
    section tells dense retrieval *which* topic). This is a deliberate, documented
    choice; flip `use_section_prefix` to False for a pure-text baseline ablation.
    """
    use_section_prefix = True
    text = chunk["text"]
    if use_section_prefix and chunk.get("section"):
        return f"{chunk['section']}. {text}"
    return text


def embed_corpus(config: dict, batch_size: int = 64) -> None:
    # Import here so the module can be imported (e.g. for build_embedding_text)
    # on a machine without sentence-transformers installed.
    from sentence_transformers import SentenceTransformer

    chunks = load_corpus(config)
    print(f"Loaded {len(chunks)} paragraphs from {config['corpus']['fms']}")

    model_name = config["retrieval"]["dense_model"]
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [build_embedding_text(c) for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors -> inner product = cosine
        convert_to_numpy=True,
    ).astype("float32")

    index_dir = resolve(config["paths"]["index"])
    index_dir.mkdir(parents=True, exist_ok=True)

    emb_path = index_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    write_jsonl(index_dir / "chunks.jsonl", chunks)

    print(f"Saved embeddings {embeddings.shape} -> {emb_path}")
    print(f"Saved chunk store ({len(chunks)} rows) -> {index_dir / 'chunks.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    embed_corpus(load_config(args.config), batch_size=args.batch_size)


if __name__ == "__main__":
    main()