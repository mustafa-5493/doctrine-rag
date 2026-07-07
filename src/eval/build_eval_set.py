"""
build_eval_set.py — Stage 3: build the evaluation set from FM paragraphs.

For a length-filtered, chapter-stratified sample of paragraphs, calls Claude to
draft (a) a natural "vanilla" question + concise gold answer, and (b) a
"negative" false-premise question + a correcting answer. Because each question
is generated FROM a known paragraph, the gold retrieval label (that paragraph's
chunk_id) is attached automatically — which is what makes recall@k / MRR
computable, and is the reason we generate rather than reuse an unlabeled dataset.

Outputs (JSONL, one row per question):
    data/eval/qa_pairs.jsonl      vanilla questions
    data/eval/negative_qa.jsonl   false-premise questions

Every row starts with source_verified=false. A human pass flips the ones you've
checked to true; downstream eval can then filter to verified-only if desired.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python -m src.eval.build_eval_set --n 150
    python -m src.eval.build_eval_set --n 150 --no-negative
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from src.utils.io import load_config, load_corpus, resolve, write_jsonl

# Verified current Anthropic API model string (see docs). Overridable via config.
DEFAULT_MODEL = "claude-sonnet-5"

# Only sample paragraphs with enough substance to yield a rich Q&A.
MIN_CHARS = 400

GEN_PROMPT = """You are helping build an evaluation set for a retrieval system \
over US Army doctrine (Field Manuals). Below is one numbered doctrine paragraph.

Write ONE natural question a soldier or analyst might ask whose answer is \
contained in this paragraph, and a concise, accurate gold answer drawn ONLY \
from the paragraph. Also classify the question as one of: definitional, \
procedural, cross_reference.

Return ONLY a JSON object, no prose, no markdown fences:
{{"question": "...", "gold_answer": "...", "question_type": "..."}}

Paragraph {para_id} (from {fm_id}):
\"\"\"
{text}
\"\"\""""

NEG_PROMPT = """You are helping build a ROBUSTNESS evaluation set for a US Army \
doctrine assistant. Below is one numbered doctrine paragraph.

Write ONE question that contains a FALSE PREMISE — a plausible-sounding but \
incorrect assumption about the content of this paragraph (e.g. wrong number, \
swapped term, inverted claim). Then write the correct answer, which should \
first correct the false premise, then state the accurate fact from the \
paragraph.

Return ONLY a JSON object, no prose, no markdown fences:
{{"question": "...", "gold_answer": "...", "false_premise": "..."}}

Paragraph {para_id} (from {fm_id}):
\"\"\"
{text}
\"\"\""""


def sample_paragraphs(chunks: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Length-filter, then sample stratified across chapters for coverage."""
    eligible = [c for c in chunks if c["char_len"] >= MIN_CHARS]
    by_chapter: dict[str, list[dict]] = defaultdict(list)
    for c in eligible:
        by_chapter[c["chapter"]].append(c)

    rng = random.Random(seed)
    chapters = sorted(by_chapter)
    # round-robin across chapters so no single chapter dominates the eval set
    for ch in chapters:
        rng.shuffle(by_chapter[ch])
    picked, i = [], 0
    while len(picked) < min(n, len(eligible)):
        ch = chapters[i % len(chapters)]
        if by_chapter[ch]:
            picked.append(by_chapter[ch].pop())
        i += 1
    return picked


def _parse_json(raw: str) -> dict:
    """Strip any accidental markdown fences and parse the JSON object."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
    return json.loads(s.strip())


def call_claude(client, model: str, prompt: str) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return _parse_json(text)


def build(config: dict, n: int, do_negative: bool = True) -> None:
    from anthropic import Anthropic  # imported here so logic is testable w/o SDK

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running.")

    model = config.get("judge", {}).get("gen_model", DEFAULT_MODEL)
    client = Anthropic()

    chunks = load_corpus(config)
    sample = sample_paragraphs(chunks, n)
    print(f"Sampled {len(sample)} paragraphs (>= {MIN_CHARS} chars) for eval set")

    vanilla, negative = [], []
    for i, c in enumerate(sample):
        base = {
            "gold_chunk_id": c["chunk_id"],
            "gold_para_id": c["para_id"],
            "fm_id": c["fm_id"],
            "source_verified": False,
        }
        try:
            v = call_claude(client, model,
                            GEN_PROMPT.format(**c))
            vanilla.append({"qid": f"v{i:04d}", "question": v["question"],
                            "gold_answer": v["gold_answer"],
                            "question_type": v.get("question_type", "definitional"),
                            **base})
        except Exception as e:
            print(f"  [skip vanilla {c['para_id']}] {e}")

        if do_negative:
            try:
                ng = call_claude(client, model, NEG_PROMPT.format(**c))
                negative.append({"qid": f"n{i:04d}", "question": ng["question"],
                                 "gold_answer": ng["gold_answer"],
                                 "false_premise": ng.get("false_premise", ""),
                                 "question_type": "negative", **base})
            except Exception as e:
                print(f"  [skip negative {c['para_id']}] {e}")

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(sample)} paragraphs processed")

    eval_dir = resolve(config["paths"].get("eval", "data/eval"))
    write_jsonl(eval_dir / "qa_pairs.jsonl", vanilla)
    print(f"Wrote {len(vanilla)} vanilla Q&A -> {eval_dir / 'qa_pairs.jsonl'}")
    if do_negative:
        write_jsonl(eval_dir / "negative_qa.jsonl", negative)
        print(f"Wrote {len(negative)} negative Q&A -> {eval_dir / 'negative_qa.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--no-negative", action="store_true")
    args = ap.parse_args()
    build(load_config(args.config), n=args.n, do_negative=not args.no_negative)


if __name__ == "__main__":
    main()