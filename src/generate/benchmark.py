"""
benchmark.py — Stage 5b: the consumer-GPU inference tradeoff benchmark.

*** MUST RUN ON THE ACTUAL GTX 1050 LAPTOP, NOT COLAB. ***
This script IS the "resource-constrained inference" measurement — running it
on a cloud A100 would produce meaningless numbers for this project's thesis.

For each downloaded GGUF quant level, sweeps n_gpu_layers and records:
    - VRAM used (via nvidia-smi)
    - tokens/sec (generation throughput)
    - OOM point (if a config fails to load / crashes)

Runs retrieval + generation end-to-end on a small sample of your verified
eval questions (retrieval on CPU via your existing hybrid retriever, so only
generation is being benchmarked against GPU constraints).

Usage (from repo root, on the laptop):
    python -m src.generate.benchmark --quant q4_k_m q8_0 --layers 0 8 16 -1 --n-questions 10

Requires llama-cpp-python built WITH CUDA support. Plain `pip install
llama-cpp-python` often gives a CPU-only wheel. If VRAM stays at 0MB
regardless of n_gpu_layers, reinstall with:

    Windows (PowerShell), CUDA Toolkit already installed:
        $env:CMAKE_ARGS="-DGGML_CUDA=on"
        pip install llama-cpp-python --force-reinstall --no-cache-dir
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

from src.generate.prompt import build_prompt
from src.retrieve.retriever import Retriever
from src.utils.io import load_config, resolve


def get_vram_used_mb() -> float | None:
    """Query nvidia-smi for current VRAM usage on GPU 0. None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return float(out.splitlines()[0])
    except Exception:
        return None


def gguf_path(models_dir: str, quant: str) -> Path:
    return Path(models_dir) / f"qwen2.5-3b-instruct-{quant}.gguf"


def run_one_config(model_path: Path, n_gpu_layers: int, questions: list[dict],
                   retriever: Retriever, top_k: int = 5,
                   max_tokens: int = 256) -> dict:
    """Load one (quant, n_gpu_layers) config, run the sample questions,
    return aggregate timing/VRAM/quality-input stats. Handles OOM/load
    failure by returning a row marked ok=False rather than crashing the
    whole sweep."""
    from src.generate.llm_local import LocalLLM

    vram_before = get_vram_used_mb()
    t_load0 = time.perf_counter()
    try:
        llm = LocalLLM(str(model_path), n_gpu_layers=n_gpu_layers)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "n_gpu_layers": n_gpu_layers}
    load_seconds = time.perf_counter() - t_load0
    vram_after_load = get_vram_used_mb()

    generations = []
    for q in questions:
        hits = retriever.search(q["question"], top_k=top_k)
        prompt = build_prompt(q["question"], hits)
        try:
            res = llm.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            generations.append({"qid": q["qid"], "ok": False, "error": str(e)[:200]})
            continue
        generations.append({
            "qid": q["qid"], "ok": True, "answer": res.text,
            "retrieved_chunk_ids": [h["chunk_id"] for h in hits],
            "tokens_per_sec": res.tokens_per_sec,
            "completion_tokens": res.completion_tokens,
        })

    vram_peak = get_vram_used_mb()
    llm.close()

    ok_gens = [g for g in generations if g["ok"]]
    tps = [g["tokens_per_sec"] for g in ok_gens]
    avg_tps = sum(tps) / len(tps) if tps else 0.0

    return {
        "ok": True,
        "n_gpu_layers": n_gpu_layers,
        "load_seconds": round(load_seconds, 2),
        "vram_before_mb": vram_before,
        "vram_after_load_mb": vram_after_load,
        "vram_peak_mb": vram_peak,
        "n_questions": len(questions),
        "n_succeeded": len(ok_gens),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "generations": generations,
    }


def sweep(config: dict, quants: list[str], layer_options: list[int],
          n_questions: int, models_dir: str = "models") -> list[dict]:
    eval_dir = resolve(config["paths"].get("eval", "data/eval"))
    questions = [json.loads(l) for l in
                open(eval_dir / "qa_pairs_verified.jsonl",
                    encoding="utf-8")][:n_questions]
    print(f"Benchmarking with {len(questions)} sample questions")

    retriever = Retriever.from_config(config)
    retriever.method = "hybrid"

    rows = []
    for quant in quants:
        mpath = gguf_path(models_dir, quant)
        if not mpath.exists():
            print(f"  [skip] {mpath} not found — run download_gguf.py first")
            continue
        for layers in layer_options:
            print(f"\n=== quant={quant}  n_gpu_layers={layers} ===")
            res = run_one_config(mpath, layers, questions, retriever)
            res["quant"] = quant
            if res["ok"]:
                print(f"  load={res['load_seconds']}s  "
                     f"VRAM_peak={res['vram_peak_mb']}MB  "
                     f"avg_tok/s={res['avg_tokens_per_sec']}  "
                     f"({res['n_succeeded']}/{res['n_questions']} ok)")
            else:
                print(f"  FAILED: {res['error']}")
            rows.append(res)

    return rows


def save_results(rows: list[dict], config: dict) -> None:
    results_dir = resolve(config["paths"].get("results", "results"))
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # summary CSV (one row per config, no per-question detail)
    csv_path = tables_dir / "benchmark_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quant", "n_gpu_layers", "ok", "load_seconds",
                   "vram_peak_mb", "avg_tokens_per_sec", "n_succeeded",
                   "n_questions", "error"])
        for r in rows:
            w.writerow([
                r.get("quant"), r.get("n_gpu_layers"), r.get("ok"),
                r.get("load_seconds", ""), r.get("vram_peak_mb", ""),
                r.get("avg_tokens_per_sec", ""), r.get("n_succeeded", ""),
                r.get("n_questions", ""), r.get("error", ""),
            ])
    print(f"\nSaved summary -> {csv_path}")

    # full detail (generations included) for later quality judging
    full_path = results_dir / "benchmark_full.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved full detail (for judge scoring) -> {full_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--quant", nargs="+", default=["q4_k_m", "q8_0"])
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 8, 16, -1],
                    help="-1 means offload all layers")
    ap.add_argument("--n-questions", type=int, default=10)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    config = load_config(args.config)
    rows = sweep(config, quants=args.quant, layer_options=args.layers,
                n_questions=args.n_questions, models_dir=args.models_dir)
    save_results(rows, config)


if __name__ == "__main__":
    main()
