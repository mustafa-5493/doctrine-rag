"""
retriever.py — Stage 3: retrieve top-k doctrine paragraphs for a query.

Loads the artifacts built by embed.py / build_faiss.py and exposes a single
Retriever class supporting three methods (set in config.retrieval.method):

    "bm25"    lexical only        (great for acronyms/jargon: METT-TC, RDSP)
    "dense"   embeddings only     (great for paraphrased / semantic queries)
    "hybrid"  fuse both via RRF   (default; robust across query types)

Hybrid uses Reciprocal Rank Fusion (RRF) rather than raw-score mixing: BM25
scores and cosine scores live on different, non-comparable scales, so fusing
by *rank* (1/(k+rank)) avoids having to normalize two incompatible score
distributions. This is the standard, defensible choice for a writeup.

The retriever is index-only and CPU-friendly — it runs on the laptop.

Example:
    from src.retrieve.retriever import Retriever
    r = Retriever.from_config()
    for hit in r.search("What is the RDSP?", top_k=5):
        print(hit["para_id"], hit["score"], hit["text"][:80])
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import faiss
import numpy as np

from src.index.build_faiss import tokenize
from src.utils.io import load_config, read_jsonl, resolve

# BGE models expect this instruction prefixed to *queries* (not passages) for
# best retrieval quality. Passages were embedded without it in embed.py.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# RRF damping constant. 60 is the value from the original RRF paper and the
# common default; larger = flatter contribution from top ranks.
RRF_K = 60


@dataclass
class Retriever:
    chunks: list[dict]
    faiss_index: faiss.Index
    bm25: object
    dense_model_name: str
    method: str = "hybrid"
    _encoder: object = None  # lazily loaded SentenceTransformer

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: dict | None = None) -> "Retriever":
        config = config or load_config()
        idx = resolve(config["paths"]["index"])
        chunks = read_jsonl(idx / "chunks.jsonl")
        faiss_index = faiss.read_index(str(idx / "faiss.index"))
        with open(idx / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        if faiss_index.ntotal != len(chunks):
            raise ValueError(
                f"Index/chunk mismatch: {faiss_index.ntotal} vs {len(chunks)}. "
                "Rebuild indexes (embed.py + build_faiss.py)."
            )
        return cls(
            chunks=chunks,
            faiss_index=faiss_index,
            bm25=bm25,
            dense_model_name=config["retrieval"]["dense_model"],
            method=config["retrieval"].get("method", "hybrid"),
        )

    # ── per-method ranked candidate lists (return list of chunk indices) ──
    def _bm25_rank(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return np.argsort(scores)[::-1][:n].tolist()

    def _dense_rank(self, query: str, n: int) -> list[int]:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.dense_model_name)
        q = self._encoder.encode(
            [BGE_QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        _, idx = self.faiss_index.search(q, n)
        return idx[0].tolist()

    @staticmethod
    def _rrf(rank_lists: list[list[int]], top_k: int) -> list[tuple[int, float]]:
        """Fuse several ranked lists of chunk indices via Reciprocal Rank Fusion."""
        fused: dict[int, float] = {}
        for ranking in rank_lists:
            for rank, doc_idx in enumerate(ranking):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── public API ───────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k chunk dicts, each with an added 'score' and 'rank'."""
        pool = max(top_k * 4, 20)  # over-retrieve per method before fusing

        if self.method == "bm25":
            hits = [(i, None) for i in self._bm25_rank(query, top_k)]
        elif self.method == "dense":
            hits = [(i, None) for i in self._dense_rank(query, top_k)]
        elif self.method == "hybrid":
            hits = self._rrf(
                [self._bm25_rank(query, pool), self._dense_rank(query, pool)],
                top_k,
            )
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

        results = []
        for rank, (doc_idx, score) in enumerate(hits):
            hit = dict(self.chunks[doc_idx])
            hit["rank"] = rank
            hit["score"] = float(score) if score is not None else None
            results.append(hit)
        return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--method", choices=["bm25", "dense", "hybrid"])
    args = ap.parse_args()

    r = Retriever.from_config()
    if args.method:
        r.method = args.method
    for hit in r.search(args.query, top_k=args.top_k):
        s = f"{hit['score']:.4f}" if hit["score"] is not None else "  -  "
        print(f"[{hit['fm_id']} {hit['para_id']}] {s}  {hit['text'][:90]}")