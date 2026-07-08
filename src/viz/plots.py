"""
plots.py — Stage 7: generate the figures for the writeup.

Reads the result artifacts produced by earlier stages and writes publication-
style PNGs to results/figures/:

  1. offload_valley.png      tok/s and VRAM vs GPU-layer offload (the PCIe
                             transfer "valley" + linear memory climb)
  2. quality_tradeoff.png    Q4 vs Q8 quality with 95% CIs + significance stars
  3. retrieval_comparison.png recall@k curves for BM25 / dense / hybrid
  4. offload_neutral.png     quality flat across offload configs (Friedman)

Each figure is self-contained and skipped gracefully if its source file is
missing, so partial result sets still produce whatever figures they can.

Usage:
    python -m src.viz.plots
"""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from src.utils.io import resolve

# ── house style ──────────────────────────────────────────────────────
Q4_COLOR = "#2b6cb0"   # blue
Q8_COLOR = "#dd6b20"   # orange
METHOD_COLORS = {"bm25": "#718096", "dense": "#2b6cb0", "hybrid": "#dd6b20"}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def _layer_label(v: int) -> str:
    return "all" if int(v) == -1 else str(int(v))


def read_csv(path) -> list[dict]:
    p = resolve(path)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def _fig_dir():
    d = resolve("results/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Figure 1: the offload valley ─────────────────────────────────────
def plot_offload_valley():
    try:
        rows = read_csv("results/tables/benchmark_summary.csv")
    except FileNotFoundError:
        print("  [skip] benchmark_summary.csv not found")
        return
    rows = [r for r in rows if r.get("ok", "True") in ("True", "true", "1")]

    def series(quant, field):
        rs = sorted((r for r in rows if r["quant"] == quant),
                    key=lambda r: (int(r["n_gpu_layers"]) == -1,
                                   int(r["n_gpu_layers"])))
        # put -1 (all) last
        order = sorted(rs, key=lambda r: (int(r["n_gpu_layers"]) if
                       int(r["n_gpu_layers"]) != -1 else 10**6))
        labels = [_layer_label(r["n_gpu_layers"]) for r in order]
        vals = [float(r[field]) for r in order]
        return labels, vals

    quants = sorted({r["quant"] for r in rows})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

    for quant in quants:
        color = Q4_COLOR if "q4" in quant.lower() else Q8_COLOR
        labels, tps = series(quant, "avg_tokens_per_sec")
        ax1.plot(labels, tps, "o-", color=color, label=quant, linewidth=2,
                 markersize=7)
        _, vram = series(quant, "vram_peak_mb")
        ax2.plot(labels, vram, "s--", color=color, label=quant, linewidth=2,
                 markersize=6)

    ax1.set_ylabel("throughput (tokens/sec)")
    ax1.set_title("Partial GPU offload is slower than none\n"
                  "(mid-range configs pay PCIe transfer cost without full "
                  "GPU compute)", fontsize=11.5, loc="left")
    ax1.legend(title="quant", frameon=False)

    ax2.axhline(4096, color="#c53030", linestyle=":", linewidth=1.2,
                label="4GB VRAM ceiling")
    ax2.set_ylabel("peak VRAM (MB)")
    ax2.set_xlabel("GPU layers offloaded")
    ax2.legend(frameon=False)

    fig.tight_layout()
    out = _fig_dir() / "offload_valley.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 2: quality tradeoff with CIs + significance ───────────────
def plot_quality_tradeoff():
    try:
        stats = load_json("results/tables/statistics.json")
    except FileNotFoundError:
        print("  [skip] statistics.json not found")
        return
    ci = stats.get("bootstrap_ci", {})
    cmp = stats.get("quant_comparison", {})

    # locate the full-offload config label for each quant
    def full_label(prefix):
        for k in ci:
            if k.startswith(prefix) and (k.endswith("@-1") or "@" not in k):
                return k
        return None
    q4 = full_label("q4")
    q8 = full_label("q8")
    if not q4 or not q8:
        print("  [skip] could not find q4/q8 configs in statistics.json")
        return

    def sig_stars(p):
        if p is None:
            return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
            else "(n.s.)"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # left: correctness + faithfulness (1-5 scale)
    ord_metrics = ["correctness", "faithfulness"]
    x = range(len(ord_metrics))
    w = 0.35
    for off, (quant, color) in enumerate([(q4, Q4_COLOR), (q8, Q8_COLOR)]):
        means, los, his = [], [], []
        for m in ord_metrics:
            c = ci[quant][m]
            means.append(c["mean"]); los.append(c["mean"] - c["ci_low"])
            his.append(c["ci_high"] - c["mean"])
        axL.bar([i + (off - 0.5) * w for i in x], means, w,
                yerr=[los, his], capsize=4, color=color,
                label=quant.split("@")[0], alpha=0.9)
    axL.set_xticks(list(x)); axL.set_xticklabels(ord_metrics)
    axL.set_ylim(3.5, 5.15); axL.set_ylabel("judge score (1-5)")
    axL.legend(frameon=False, title="quant")
    for i, m in enumerate(ord_metrics):
        p = cmp.get(m, {}).get("p")
        axL.text(i, 5.02, sig_stars(p), ha="center", fontsize=11)

    # right: citation accuracy (0-1)
    cm = "cited_gold_paragraph"
    means, los, his = [], [], []
    for quant in (q4, q8):
        c = ci[quant][cm]
        means.append(c["mean"]); los.append(c["mean"] - c["ci_low"])
        his.append(c["ci_high"] - c["mean"])
    axR.bar([0, 1], means, 0.55, yerr=[los, his], capsize=4,
            color=[Q4_COLOR, Q8_COLOR], alpha=0.9)
    axR.set_xticks([0, 1]); axR.set_xticklabels([q4.split("@")[0], q8.split("@")[0]])
    axR.set_ylim(0, 1.0); axR.set_ylabel("citation accuracy")
    p = cmp.get(cm, {}).get("p")
    axR.text(0.5, max(means) + 0.12, sig_stars(p), ha="center", fontsize=11)
    axR.set_title("citation", fontsize=11)

    fig.suptitle("8-bit vs 4-bit quantization: quality with 95% CIs "
                 "(*=p<0.05, n.s.=not significant)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _fig_dir() / "quality_tradeoff.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 3: retrieval comparison ───────────────────────────────────
def plot_retrieval():
    try:
        rows = read_csv("results/tables/retrieval_metrics.csv")
    except FileNotFoundError:
        print("  [skip] retrieval_metrics.csv not found")
        return
    ks = [c for c in rows[0] if c.startswith("recall@")]
    kvals = [int(c.split("@")[1]) for c in ks]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        method = r["method"]
        recalls = [float(r[c]) for c in ks]
        ax.plot(kvals, recalls, "o-", label=f"{method} (MRR={float(r['mrr']):.3f})",
                color=METHOD_COLORS.get(method, "#333"), linewidth=2, markersize=7)
    ax.set_xticks(kvals)
    ax.set_xlabel("k"); ax.set_ylabel("recall@k")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Hybrid retrieval outperforms lexical and dense alone\n"
                 "(FM 5-0, 150 verified questions)", fontsize=11.5, loc="left")
    ax.legend(frameon=False, title="method")
    fig.tight_layout()
    out = _fig_dir() / "retrieval_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 4: offload is quality-neutral ─────────────────────────────
def plot_offload_neutral():
    try:
        rows = read_csv("results/tables/quality_by_quant.csv")
    except FileNotFoundError:
        print("  [skip] quality_by_quant.csv not found")
        return
    # only meaningful if we have per-offload configs (labels with '@')
    if not any("@" in r["config"] for r in rows):
        print("  [skip] quality_by_quant.csv has no per-offload configs")
        return

    def parse(label):
        q, l = label.rsplit("@", 1)
        return q, int(l)

    quants = sorted({parse(r["config"])[0] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for quant in quants:
        color = Q4_COLOR if "q4" in quant.lower() else Q8_COLOR
        rs = sorted((r for r in rows if parse(r["config"])[0] == quant),
                    key=lambda r: (parse(r["config"])[1] if
                                   parse(r["config"])[1] != -1 else 10**6))
        labels = [_layer_label(parse(r["config"])[1]) for r in rs]
        vals = [float(r["mean_correctness"]) for r in rs]
        ax.plot(labels, vals, "o-", color=color, label=quant, linewidth=2,
                markersize=7)
    ax.set_xlabel("GPU layers offloaded")
    ax.set_ylabel("mean correctness (1-5)")
    ax.set_ylim(3.8, 4.6)
    ax.set_title("Offload configuration does not affect quality\n"
                 "(flat within quant level; Friedman n.s.)", fontsize=11.5,
                 loc="left")
    ax.legend(frameon=False, title="quant")
    fig.tight_layout()
    out = _fig_dir() / "offload_neutral.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Generating figures -> results/figures/")
    plot_offload_valley()
    plot_quality_tradeoff()
    plot_retrieval()
    plot_offload_neutral()
    print("Done.")


if __name__ == "__main__":
    main()
