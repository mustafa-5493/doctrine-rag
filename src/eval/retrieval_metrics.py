"""
retrieval_metrics.py — Stage 4: retrieval quality metrics.

Runs the verified vanilla eval set through the retriever and computes, for
each retrieval method (bm25 / dense / hybrid):

    recall@k   fraction of questions where the gold paragraph appears
               anywhere in the top-k results (reported for several k)
    MRR        mean reciprocal rank of the gold paragraph (0 if absent
               from the searched pool)

This is the retrieval-half of your ablation table — the "does hybrid beat
BM25/dense alone on doctrine text" result. Negative-split questions are
excluded from these metrics by default: their gold answer is a *correction*,
not a simple paragraph lookup, so recall/MRR against gold_chunk_id is still
meaningful (the corrected fact does live in that paragraph) but you may want
to report them separately — see --include-negative.

Usage:
    python -m src.eval.retrieval_metrics
    python -m src.eval.retrieval_metrics --k 1 3 5 10 --include-negative
"""

from __future__ import annotations

import argparse
import json

from src.retrieve.retriever import Retriever
from src.utils.io import load_config, resolve


def load_eval_questions(config: dict, include_negative: bool = False) -> list[dict]:
    eval_dir = resolve(config["paths"].get("eval", "data/eval"))
    qs = [json.loads(l) for l in open(eval_dir / "qa_pairs_verified.jsonl",
                                      encoding="utf-8")]
    for q in qs:
        q["_split"] = "vanilla"
    if include_negative:
        neg_path = eval_dir / "negative_qa_verified.jsonl"
        if neg_path.exists():
            neg = [json.loads(l) for l in open(neg_path, encoding="utf-8")]
            for q in neg:
                q["_split"] = "negative"
            qs += neg
    return qs


def evaluate_method(retriever: Retriever, questions: list[dict],
                    ks: list[int]) -> dict:
    """Compute recall@k (for each k) and MRR for one retrieval method."""
    max_k = max(ks)
    ranks: list[int | None] = []  # 1-indexed rank of gold chunk, None if absent

    for q in questions:
        hits = retriever.search(q["question"], top_k=max_k)
        hit_ids = [h["chunk_id"] for h in hits]
        try:
            rank = hit_ids.index(q["gold_chunk_id"]) + 1  # 1-indexed
        except ValueError:
            rank = None
        ranks.append(rank)

    n = len(ranks)
    recall_at_k = {
        k: sum(1 for r in ranks if r is not None and r <= k) / n
        for k in ks
    }
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    return {"recall": recall_at_k, "mrr": mrr, "n": n, "ranks": ranks}


def run(config: dict, ks: list[int], include_negative: bool = False) -> dict:
    questions = load_eval_questions(config, include_negative=include_negative)
    print(f"Loaded {len(questions)} eval questions "
          f"({'vanilla + negative' if include_negative else 'vanilla only'})")

    results = {}
    for method in ["bm25", "dense", "hybrid"]:
        retriever = Retriever.from_config(config)
        retriever.method = method
        res = evaluate_method(retriever, questions, ks)
        results[method] = res
        recall_str = "  ".join(f"R@{k}={res['recall'][k]:.3f}" for k in ks)
        print(f"[{method:6s}] n={res['n']}  {recall_str}  MRR={res['mrr']:.3f}")

    return results


def save_results(results: dict, config: dict, ks: list[int]) -> None:
    """Write a CSV ablation table to results/tables/ (created if missing)."""
    import csv
    out_dir = resolve(config["paths"].get("results", "results")) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "retrieval_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", *[f"recall@{k}" for k in ks], "mrr", "n"])
        for method, res in results.items():
            w.writerow([method, *[f"{res['recall'][k]:.4f}" for k in ks],
                       f"{res['mrr']:.4f}", res["n"]])
    print(f"Saved ablation table -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--include-negative", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    results = run(config, ks=args.k, include_negative=args.include_negative)
    save_results(results, config, ks=args.k)


if __name__ == "__main__":
    main()
