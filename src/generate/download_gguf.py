"""
download_gguf.py — Stage 5a: fetch pre-quantized GGUF weights.

Uses Qwen's official pre-quantized releases rather than converting HF weights
ourselves (which requires cloning/building llama.cpp) — same model, already
converted, one file per quant level.

Can run anywhere with network access (Colab or laptop) — this is just a file
download, no GPU compute happens here. The files then need to be present on
whichever machine runs benchmark.py (your GTX 1050 laptop).

Usage:
    python -m src.generate.download_gguf --quant q4_k_m q8_0
    python -m src.generate.download_gguf --quant q4_k_m q8_0 --out-dir models/
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"

# Confirmed available quant levels for this repo (mirrors the sibling
# Qwen2.5-*-Instruct-GGUF repos' naming convention).
FILENAME_TEMPLATE = "qwen2.5-3b-instruct-{quant}.gguf"
VALID_QUANTS = ["q2_k", "q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m",
                "q6_k", "q8_0"]


def download(quants: list[str], out_dir: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for q in quants:
        q = q.lower()
        if q not in VALID_QUANTS:
            raise ValueError(f"Unknown quant '{q}'. Valid: {VALID_QUANTS}")
        fname = FILENAME_TEMPLATE.format(quant=q)
        print(f"Downloading {fname} ...")
        path = hf_hub_download(repo_id=REPO_ID, filename=fname,
                               local_dir=str(out))
        paths.append(Path(path))
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  -> {path} ({size_mb:.0f} MB)")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", nargs="+", default=["q4_k_m", "q8_0"],
                    help=f"quant levels to download, from {VALID_QUANTS}")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()
    download(args.quant, args.out_dir)


if __name__ == "__main__":
    main()
