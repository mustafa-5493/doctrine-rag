"""
judge.py — Stage 5c: score generation quality with Claude as an independent judge.

Reads the generations produced by benchmark.py (results/benchmark_full.jsonl)
and scores each answer against (a) the human-verified gold answer [correctness]
and (b) the retrieved passages [faithfulness], plus whether it cited the gold
paragraph. Uses Claude (Anthropic API) — a different model family from the
Qwen generator, so there's no self-grading bias.

Key efficiency + validity point: generation ran at temperature 0 (deterministic),
and n_gpu_layers only changes WHERE compute happens, not the output text. So
quality depends on the QUANT LEVEL, not the offload config. This script judges
one representative config per quant level (default: full offload, n_gpu_layers=-1)
and separately reports how identical the outputs are across offload configs —
which both saves ~5x the API calls and supports the claim that offload is a
pure speed/memory knob, orthogonal to quality.

Outputs:
    results/tables/quality_by_quant.csv   mean correctness/faithfulness/citation
    results/quality_detail.jsonl          per-question judge scores + reasoning

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python -m src.eval.judge
    python -m src.eval.judge --all-configs      # judge every config (5x cost)
    python -m src.eval.judge --n 50             # cap questions per config
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from src.eval.build_eval_set import call_claude
from src.utils.io import load_config, load_corpus, read_jsonl, resolve

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

JUDGE_PROMPT = """You are an impartial judge scoring an AI-generated answer to a \
question about US Army doctrine. Score against the provided gold answer and \
reference passages ONLY — do not use outside knowledge.

Score three things:
1. correctness (1-5): how well the generated answer matches the FACTS in the \
gold answer. 5 = all key facts correct and present; 1 = wrong or missing the point.
2. faithfulness (1-5): how well the generated answer is SUPPORTED by the \
reference passages. 5 = every claim traceable to the passages; 1 = hallucinated.
3. cited_gold_paragraph (true/false): whether the answer cites the gold \
paragraph id ({gold_para_id}).

Return ONLY JSON:
{{"correctness": <1-5>, "faithfulness": <1-5>, "cited_gold_paragraph": <bool>, \
"reason": "<=20 words"}}

QUESTION: {question}

GOLD ANSWER: {gold_answer}
GOLD PARAGRAPH ID: {gold_para_id}

REFERENCE PASSAGES:
{context}

AI-GENERATED ANSWER: {generated_answer}"""


def index_by_config(results: list[dict]) -> dict:
    """Return {(quant, n_gpu_layers): {qid: generation_dict}}."""
    out = {}
    for r in results:
        if not r.get("ok"):
            continue
        key = (r["quant"], r["n_gpu_layers"])
        out[key] = {g["qid"]: g for g in r.get("generations", []) if g.get("ok")}
    return out


def output_consistency_report(by_config: dict) -> None:
    """For each quant, report how identical answers are across offload configs.
    Free diagnostic (no API calls) — validates 'offload doesn't affect quality'."""
    by_quant = defaultdict(list)
    for (quant, layers), gens in by_config.items():
        by_quant[quant].append((layers, gens))

    print("\n--- output consistency across offload configs (per quant) ---")
    for quant, configs in by_quant.items():
        if len(configs) < 2:
            print(f"  {quant}: only one config, skipping")
            continue
        # compare every config's answers to the -1 (full offload) reference
        ref_layers = -1 if any(l == -1 for l, _ in configs) else configs[0][0]
        ref = dict(configs)[ref_layers]
        qids = list(ref)
        total, identical = 0, 0
        for layers, gens in configs:
            if layers == ref_layers:
                continue
            for qid in qids:
                if qid in gens:
                    total += 1
                    if gens[qid]["answer"].strip() == ref[qid]["answer"].strip():
                        identical += 1
        pct = 100 * identical / total if total else 0
        print(f"  {quant}: {identical}/{total} answers identical to full-offload "
              f"({pct:.1f}%) across other configs")


def representative_configs(by_config: dict) -> dict:
    """Pick one config per quant to judge: prefer n_gpu_layers=-1."""
    by_quant = defaultdict(list)
    for (quant, layers), gens in by_config.items():
        by_quant[quant].append((layers, gens))
    chosen = {}
    for quant, configs in by_quant.items():
        pick = next((g for l, g in configs if l == -1), configs[0][1])
        pick_layers = next((l for l, g in configs if l == -1), configs[0][0])
        chosen[quant] = (pick_layers, pick)
    return chosen


def judge_generations(client, model, gens: dict, questions: dict,
                      chunks: dict, cap: int | None) -> list[dict]:
    scored = []
    qids = list(gens)[:cap] if cap else list(gens)
    for i, qid in enumerate(qids):
        g = gens[qid]
        q = questions.get(qid)
        if q is None:
            continue
        context = "\n\n".join(
            f"[{chunks[cid]['fm_id']} {chunks[cid]['para_id']}]\n{chunks[cid]['text']}"
            for cid in g.get("retrieved_chunk_ids", []) if cid in chunks
        )
        prompt = JUDGE_PROMPT.format(
            question=q["question"], gold_answer=q["gold_answer"],
            gold_para_id=q["gold_para_id"], context=context,
            generated_answer=g["answer"],
        )
        try:
            v = call_claude(client, model, prompt)
            scored.append({
                "qid": qid, "gold_para_id": q["gold_para_id"],
                "correctness": v.get("correctness"),
                "faithfulness": v.get("faithfulness"),
                "cited_gold_paragraph": v.get("cited_gold_paragraph"),
                "reason": v.get("reason", ""),
            })
        except Exception as e:
            scored.append({"qid": qid, "error": str(e)[:150]})
        if (i + 1) % 25 == 0:
            print(f"    judged {i + 1}/{len(qids)}")
    return scored


def aggregate(scored: list[dict]) -> dict:
    ok = [s for s in scored if "error" not in s and s.get("correctness") is not None]
    n = len(ok)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "mean_correctness": round(sum(s["correctness"] for s in ok) / n, 3),
        "mean_faithfulness": round(sum(s["faithfulness"] for s in ok) / n, 3),
        "citation_accuracy": round(
            sum(1 for s in ok if s.get("cited_gold_paragraph")) / n, 3),
    }


def run(config: dict, all_configs: bool, cap: int | None) -> None:
    from anthropic import Anthropic  # noqa: F401

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running.")

    model = config.get("judge", {}).get("model", DEFAULT_JUDGE_MODEL)
    client = Anthropic()

    results_dir = resolve(config["paths"].get("results", "results"))
    results = read_jsonl(results_dir / "benchmark_full.jsonl")
    by_config = index_by_config(results)

    # free diagnostic
    output_consistency_report(by_config)

    # which configs to actually judge
    if all_configs:
        to_judge = {f"{q}@{l}": gens for (q, l), gens in by_config.items()}
    else:
        to_judge = {q: gens for q, (l, gens) in representative_configs(by_config).items()}

    eval_dir = resolve(config["paths"].get("eval", "data/eval"))
    questions = {r["qid"]: r for r in
                read_jsonl(eval_dir / "qa_pairs_verified.jsonl")}
    chunks = {c["chunk_id"]: c for c in load_corpus(config)}

    summary_rows, detail_rows = [], []
    for label, gens in to_judge.items():
        print(f"\n=== judging {label} ({len(gens)} generations) ===")
        scored = judge_generations(client, model, gens, questions, chunks, cap)
        agg = aggregate(scored)
        agg["config"] = label
        summary_rows.append(agg)
        for s in scored:
            s["config"] = label
            detail_rows.append(s)
        print(f"  -> correctness={agg.get('mean_correctness')} "
              f"faithfulness={agg.get('mean_faithfulness')} "
              f"citation_acc={agg.get('citation_accuracy')} (n={agg['n']})")

    # save
    import csv
    tables = results_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with open(tables / "quality_by_quant.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean_correctness", "mean_faithfulness",
                   "citation_accuracy", "n"])
        for r in summary_rows:
            w.writerow([r["config"], r.get("mean_correctness"),
                       r.get("mean_faithfulness"), r.get("citation_accuracy"),
                       r["n"]])
    with open(results_dir / "quality_detail.jsonl", "w", encoding="utf-8") as f:
        for r in detail_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved quality table -> {tables / 'quality_by_quant.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--all-configs", action="store_true",
                    help="judge every offload config (5x API cost)")
    ap.add_argument("--n", type=int, default=None,
                    help="cap questions judged per config")
    args = ap.parse_args()
    run(load_config(args.config), all_configs=args.all_configs, cap=args.n)


if __name__ == "__main__":
    main()
