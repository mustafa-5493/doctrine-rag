"""
prompt.py — assembles a RAG prompt from a query and retrieved passages.
"""

from __future__ import annotations

QWEN_TEMPLATE = """<|im_start|>system
You are a helpful assistant that answers questions about US Army doctrine \
using ONLY the provided reference passages. Cite the paragraph number(s) you \
used. If the passages don't contain the answer, say so.<|im_end|>
<|im_start|>user
Reference passages:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h['fm_id']} {h['para_id']}]\n{h['text']}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[dict]) -> str:
    return QWEN_TEMPLATE.format(context=format_context(hits), question=question)
