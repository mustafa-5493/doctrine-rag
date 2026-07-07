"""
screen_eval_set.py — Stage 3b: automated faithfulness pre-screen.

Runs each generated Q&A past Claude with ONE narrow question: is the answer
fully supported by its own source paragraph? The model is pinned to the
provided paragraph as sole ground truth (it must NOT use outside knowledge),
and told to FLAG when in doubt. This is a triage step, not a source of truth:
it decides *what a human should read*, not what is correct.

Honest-methodology design: the review file contains every FLAGGED row PLUS a
random sample of PASSED rows, so a human still spot-checks the "clean" set
rather than trusting the screen blindly. Note the screen model is the same
family as the generator, so it can share blind spots — hence the human stays
in the loop on flags and on a pass sample.

Outputs:
    data/eval/<name>_screened.jsonl   every row + screen_flag + screen_reason
    <name>_flagged_review.txt         flagged rows + pass-sample, readable blocks

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python -m src.eval.screen_eval_set --split vanilla
    python -m src.eval.screen_eval_set --split negative
"""

from __future__ import annotations

import argparse
import json
import os
import random

from src.eval.build_eval_set import DEFAULT_MODEL, call_claude
from src.utils.io import load_config, load_corpus, resolve

VANILLA_SCREEN = """You are screening one Q&A pair for an evaluation set. Judge \
ONLY against the paragraph below — do NOT use any outside knowledge of Army \
doctrine. Your question: is EVERY claim in the answer supported by this \
paragraph?

Return ONLY JSON: {{"verdict": "PASS" or "FLAG", "reason": "<=15 words"}}
- PASS: every claim in the answer appears in / follows from the paragraph.
- FLAG: the answer contains anything not supported by the paragraph, OR the \
question isn't answerable from it. When uncertain, FLAG.

PARAGRAPH ({para_id}):
\"\"\"{source}\"\"\"

QUESTION: {question}
ANSWER: {answer}"""

NEGATIVE_SCREEN = """You are screening one FALSE-PREMISE Q&A pair for a \
robustness eval. Judge ONLY against the paragraph below — no outside knowledge.

A valid item must satisfy BOTH: (1) the QUESTION contains a false premise \
(a claim that contradicts the paragraph), and (2) the ANSWER corrects that \
false premise and then states what the paragraph actually says.

Return ONLY JSON: {{"verdict": "PASS" or "FLAG", "reason": "<=15 words"}}
- PASS: both conditions hold.
- FLAG: the question has no real false premise, or the answer fails to correct \
it, or the corrected fact isn't in the paragraph. When uncertain, FLAG.

PARAGRAPH ({para_id}):
\"\"\"{source}\"\"\"

QUESTION: {question}
ANSWER: {answer}"""


def screen(config: dict, split: str, pass_sample_frac: float = 0.2,
           seed: int = 0) -> None:
    from anthropic import Anthropic  # noqa: F401 (import guarded for testability)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("Set ANTHROPIC_API_KEY before running.")

    model = config.get("judge", {}).get("gen_model", DEFAULT_MODEL)
    client = Anthropic()

    eval_dir = resolve(config["paths"].get("eval", "data/eval"))
    fname = "qa_pairs" if split == "vanilla" else "negative_qa"
    rows = [json.loads(l) for l in
           open(eval_dir / f"{fname}.jsonl", encoding="utf-8")]

    chunks = {c["chunk_id"]: c for c in load_corpus(config)}
    template = VANILLA_SCREEN if split == "vanilla" else NEGATIVE_SCREEN

    flagged, passed = [], []
    for i, r in enumerate(rows):
        src = chunks[r["gold_chunk_id"]]["text"]
        prompt = template.format(para_id=r["gold_para_id"], source=src,
                                 question=r["question"], answer=r["gold_answer"])
        try:
            v = call_claude(client, model, prompt)
            r["screen_flag"] = v.get("verdict", "FLAG").upper()
            r["screen_reason"] = v.get("reason", "")
        except Exception as e:
            r["screen_flag"] = "FLAG"
            r["screen_reason"] = f"screen error: {e}"
        (flagged if r["screen_flag"] == "FLAG" else passed).append(r)
        if (i + 1) % 25 == 0:
            print(f"  ... screened {i + 1}/{len(rows)}")

    # persist full screened set
    out_jsonl = eval_dir / f"{fname}_screened.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in flagged + passed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # build readable review file: all flagged + a random pass-sample
    rng = random.Random(seed)
    sample = rng.sample(passed, max(1, int(len(passed) * pass_sample_frac))) \
        if passed else []
    review_rows = [("FLAG", r) for r in flagged] + [("PASS-CHECK", r) for r in sample]
    review_path = f"{fname}_flagged_review.txt"
    with open(review_path, "w", encoding="utf-8") as f:
        for tag, r in review_rows:
            src = chunks[r["gold_chunk_id"]]["text"]
            f.write(f"{'='*70}\n[{tag}] qid={r['qid']} para={r['gold_para_id']} "
                    f"reason={r.get('screen_reason','')}\n{'-'*70}\n")
            f.write(f"Q: {r['question']}\nA: {r['gold_answer']}\n")
            if r.get("false_premise"):
                f.write(f"FALSE PREMISE: {r['false_premise']}\n")
            f.write(f"\nSOURCE ({r['gold_para_id']}):\n{src}\n\n"
                    f"DROP? (note qid if bad): \n\n")

    print(f"\n{split}: {len(rows)} screened | {len(flagged)} FLAGGED | "
          f"{len(passed)} passed")
    print(f"Review file: {review_path} "
          f"({len(flagged)} flagged + {len(sample)} pass-checks to read)")
    print(f"Full screened set: {out_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--split", choices=["vanilla", "negative"], default="vanilla")
    ap.add_argument("--pass-sample", type=float, default=0.2,
                    help="fraction of PASSed rows to include for human spot-check")
    args = ap.parse_args()
    screen(load_config(args.config), split=args.split,
           pass_sample_frac=args.pass_sample)


if __name__ == "__main__":
    main()
