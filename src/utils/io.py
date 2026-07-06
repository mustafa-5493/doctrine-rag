"""
io.py — shared IO utilities for the Doctrine-RAG pipeline.

Keeps path/config handling in one place so notebooks and scripts stay thin.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


# ── Repo-root resolution ─────────────────────────────────────────────
# This file lives at <repo>/src/utils/io.py, so the repo root is 3 parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load the YAML config. Relative paths are resolved from the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    path = resolve(path)
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write a list of dicts to JSONL, creating parent dirs as needed."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fm_to_filename(fm_id: str) -> str:
    """'FM 5-0' -> 'fm5-0.jsonl' (matches fm_parser.py output naming)."""
    return fm_id.lower().replace("fm ", "fm").replace(" ", "") + ".jsonl"


def load_corpus(config: dict) -> list[dict]:
    """
    Load and concatenate parsed paragraphs for every FM listed in
    config['corpus']['fms']. Returns a flat list of chunk dicts, each
    tagged with a stable unique 'chunk_id' of the form '<FM>::<para_id>'.
    """
    parsed_dir = Path(config["paths"]["parsed"])
    chunks: list[dict] = []
    for fm_id in config["corpus"]["fms"]:
        fpath = parsed_dir / fm_to_filename(fm_id)
        recs = read_jsonl(fpath)
        for r in recs:
            r["chunk_id"] = f"{r['fm_id']}::{r['para_id']}"
            chunks.append(r)
    return chunks