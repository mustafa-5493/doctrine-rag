"""
stats.py — Stage 6: statistical robustness analysis of the quality results.

Runs entirely on results/quality_detail.jsonl (the per-question judge scores) —
no API calls. Turns point estimates into estimates-with-uncertainty and tests
whether the observed differences are real or noise.

Produces three things:

1. Bootstrap 95% CIs for every (config, metric) mean. Answers "how precise is
   each number?" using 10k resamples of the per-question scores.

2. Paired Q4-vs-Q8 tests (at full offload, n_gpu_layers=-1). Same questions
   scored under both quant levels, so pairing is valid:
     - correctness / faithfulness (ordinal 1-5): Wilcoxon signed-rank
     - citation (binary): McNemar exact test
   plus a bootstrap CI on the paired mean difference (the most interpretable
   "is it significant" signal — does the difference CI exclude zero?).

3. Friedman test across offload configs within each quant. This is the
   repeated-measures (same questions, multiple conditions) non-parametric test.
   A non-significant result formally backs the "offload is quality-neutral"
   claim rather than eyeballing the spread.

Usage:
    python -m src.eval.stats
    python -m src.eval.stats --n-boot 20000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats as sp

METRICS = ["correctness", "faithfulness", "cited_gold_paragraph"]


def load_detail(path) -> list[dict]:
    from src.utils.io import resolve
    p = resolve(path)
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or r.get("correctness") is None:
                continue
            rows.append(r)
    return rows


def parse_config(label: str) -> tuple[str, int | None]:
    """'q4_k_m@-1' -> ('q4_k_m', -1); 'q4_k_m' -> ('q4_k_m', None)."""
    if "@" in label:
        quant, layers = label.rsplit("@", 1)
        return quant, int(layers)
    return label, None


def metric_value(row: dict, metric: str) -> float:
    v = row[metric]
    if metric == "cited_gold_paragraph":
        return 1.0 if v else 0.0
    return float(v)


def pivot(rows: list[dict], config: str, metric: str) -> dict[str, float]:
    """{qid: score} for one config+metric."""
    return {r["qid"]: metric_value(r, metric)
            for r in rows if r["config"] == config}


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 ci: float = 95.0, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.array([rng.choice(values, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(boots, (100 - ci) / 2)
    hi = np.percentile(boots, 100 - (100 - ci) / 2)
    return float(values.mean()), float(lo), float(hi)


def paired_common(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Aligned score arrays over the qids present in both configs."""
    qids = sorted(set(a) & set(b))
    return (np.array([a[q] for q in qids]),
            np.array([b[q] for q in qids]))


def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> dict:
    diff = x - y
    if np.all(diff == 0):
        return {"test": "wilcoxon", "p": 1.0, "note": "all differences zero"}
    try:
        stat, p = sp.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return {"test": "wilcoxon", "statistic": float(stat), "p": float(p)}
    except ValueError as e:
        return {"test": "wilcoxon", "p": None, "note": str(e)[:80]}


def mcnemar_exact(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired binary test. x,y in {0,1}. Uses exact binomial on discordant pairs."""
    b = int(np.sum((x == 1) & (y == 0)))  # x hit, y miss
    c = int(np.sum((x == 0) & (y == 1)))  # x miss, y hit
    n_disc = b + c
    if n_disc == 0:
        return {"test": "mcnemar_exact", "p": 1.0, "b": b, "c": c,
                "note": "no discordant pairs"}
    p = sp.binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue
    return {"test": "mcnemar_exact", "p": float(p), "b": b, "c": c}


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                      seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI on the paired mean difference (x - y)."""
    diff = x - y
    return bootstrap_ci(diff, n_boot=n_boot, seed=seed)


def friedman_across_offload(rows: list[dict], quant: str, metric: str) -> dict:
    """Repeated-measures test across offload configs within one quant."""
    configs = sorted({r["config"] for r in rows
                      if parse_config(r["config"])[0] == quant})
    if len(configs) < 3:
        return {"test": "friedman", "p": None,
                "note": f"only {len(configs)} configs (need >=3)"}
    piv = {c: pivot(rows, c, metric) for c in configs}
    common = set.intersection(*[set(p) for p in piv.values()])
    if len(common) < 3:
        return {"test": "friedman", "p": None, "note": "too few common qids"}
    common = sorted(common)
    arrays = [np.array([piv[c][q] for q in common]) for c in configs]
    # Friedman requires variation; if every condition identical it errors
    try:
        stat, p = sp.friedmanchisquare(*arrays)
        return {"test": "friedman", "statistic": float(stat), "p": float(p),
                "k_configs": len(configs), "n_questions": len(common)}
    except ValueError as e:
        return {"test": "friedman", "p": None, "note": str(e)[:80]}


def run(detail_path: str, n_boot: int) -> None:
    from src.utils.io import resolve

    rows = load_detail(detail_path)
    configs = sorted({r["config"] for r in rows})
    quants = sorted({parse_config(c)[0] for c in configs})
    print(f"Loaded {len(rows)} scored rows across {len(configs)} configs "
          f"({', '.join(configs)})\n")

    report = {"bootstrap_ci": {}, "quant_comparison": {}, "offload_friedman": {}}

    # --- 1. Bootstrap CIs per config/metric ---
    print("=" * 68)
    print("1. BOOTSTRAP 95% CONFIDENCE INTERVALS (per config, per metric)")
    print("=" * 68)
    for config in configs:
        report["bootstrap_ci"][config] = {}
        line = f"  {config:14s}"
        for metric in METRICS:
            vals = np.array(list(pivot(rows, config, metric).values()))
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot)
            report["bootstrap_ci"][config][metric] = {
                "mean": round(mean, 3), "ci_low": round(lo, 3),
                "ci_high": round(hi, 3), "n": len(vals)}
            short = {"correctness": "corr", "faithfulness": "faith",
                     "cited_gold_paragraph": "cite"}[metric]
            line += f"  {short}={mean:.2f}[{lo:.2f},{hi:.2f}]"
        print(line)

    # --- 2. Paired Q4 vs Q8 at full offload ---
    print("\n" + "=" * 68)
    print("2. PAIRED QUANT COMPARISON (Q4 vs Q8, full offload @-1)")
    print("=" * 68)
    # find the full-offload config label for each quant
    def full_offload_label(quant):
        cands = [c for c in configs if parse_config(c) == (quant, -1)]
        if cands:
            return cands[0]
        # representative run: bare quant label
        return quant if quant in configs else None

    if len(quants) == 2:
        qA, qB = quants  # e.g. q4_k_m, q8_0
        cA, cB = full_offload_label(qA), full_offload_label(qB)
        if cA and cB:
            for metric in METRICS:
                a = pivot(rows, cA, metric)
                b = pivot(rows, cB, metric)
                x, y = paired_common(a, b)
                if metric == "cited_gold_paragraph":
                    test = mcnemar_exact(x, y)
                else:
                    test = wilcoxon_paired(x, y)
                mdiff, dlo, dhi = bootstrap_diff_ci(x, y, n_boot=n_boot)
                sig = ("SIG" if test.get("p") is not None and test["p"] < 0.05
                       else "n.s.")
                report["quant_comparison"][metric] = {
                    **test, "mean_diff": round(mdiff, 3),
                    "diff_ci": [round(dlo, 3), round(dhi, 3)], "n": len(x)}
                pstr = f"p={test['p']:.4f}" if test.get("p") is not None else "p=NA"
                print(f"  {metric:22s} {qA}-{qB} diff={mdiff:+.3f} "
                      f"CI[{dlo:+.3f},{dhi:+.3f}]  {test['test']} {pstr}  [{sig}]")
    else:
        print(f"  Need exactly 2 quant levels, found {len(quants)}: skipping")

    # --- 3. Friedman across offload configs within each quant ---
    print("\n" + "=" * 68)
    print("3. OFFLOAD NEUTRALITY (Friedman test across offload configs)")
    print("=" * 68)
    for quant in quants:
        report["offload_friedman"][quant] = {}
        for metric in METRICS:
            res = friedman_across_offload(rows, quant, metric)
            report["offload_friedman"][quant][metric] = res
            if res.get("p") is not None:
                verdict = ("configs DIFFER (p<0.05)" if res["p"] < 0.05
                           else "no sig. difference -> offload is quality-neutral")
                print(f"  {quant:8s} {metric:22s} "
                      f"chi2={res.get('statistic', float('nan')):.2f} "
                      f"p={res['p']:.4f}  {verdict}")
            else:
                print(f"  {quant:8s} {metric:22s} {res.get('note')}")

    # --- save ---
    out = resolve("results/tables/statistics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full statistics -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="results/quality_detail.jsonl")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    run(args.detail, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
