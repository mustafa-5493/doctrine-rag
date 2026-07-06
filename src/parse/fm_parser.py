"""
fm_parser.py — Paragraph-aware parser for US Army Field Manuals (APD PDFs).

Built and tested against FM 5-0 (ARN44590). Extracts doctrine-native
paragraph units (e.g. "1-137.", "3-7.") as chunks, preserving the
citable paragraph number, chapter, and nearest section header. Strips
running headers/footers and skips blank filler pages.

Designed for the Doctrine-RAG project Stage 1 (data preprocessing).
Output: one JSONL record per doctrine paragraph.

Usage:
    python fm_parser.py INPUT.pdf OUTPUT.jsonl --fm-id "FM 5-0"
"""

import argparse
import json
import re
import sys

import pdfplumber

# ---- Regex patterns (tuned to APD FM layout) --------------------------------

# Paragraph marker at line start. Two valid shapes:
#   numeric chapter:  "1-137."  "3-7."   -> \d+-\d+
#   appendix letter:  "G-3."    "A-12."  -> [A-Z]-\d+
PARA_RE = re.compile(r"^\s*((?:\d+|[A-Z])-\d+)\.\s+(.*)$")

# Footer line, either orientation:
#   "26   FM 5-0 (INCL C1)   04 November 2024"
#   "04 November 2024   FM 5-0 (INCL C1)   27"
FOOTER_RE = re.compile(
    r"(FM\s+\d+-\d+.*\d{4})|(\d{1,2}\s+\w+\s+\d{4}\s+FM\s+\d+-\d+)",
    re.IGNORECASE,
)

# A bare page number sitting alone on a line (residual footer fragment)
BARE_PAGENUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Blank filler pages
BLANK_PAGE_RE = re.compile(r"This page intentionally left blank\.?", re.IGNORECASE)

# ALL-CAPS section header (allows digits, spaces, &, hyphen, parens)
CAPS_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 &()\-/,]{3,}$")

# Cross-references like "(See FM 3-0 ...)" or "(figure 3-1 on page 48)"
XREF_RE = re.compile(r"\((?:see|for more|figure|table)[^)]*\)", re.IGNORECASE)

# Figure/table caption line, e.g. "Figure E-5. Sample Annex E ..." — marks the
# start of a sample-order TEMPLATE block (appendices D/E), which is fill-in-the-
# blank order format, not doctrine prose. Stop accumulating the paragraph here.
FIGURE_CAP_RE = re.compile(r"^\s*(Figure|Table)\s+[A-Z]?\d*-?\d+\.", re.IGNORECASE)

# Order-template boilerplate tokens that should never be inside a doctrine chunk.
TEMPLATE_NOISE_RE = re.compile(r"\[CLASSIFICATION\]|\[page number\]", re.IGNORECASE)


def strip_page_furniture(text: str, fm_id: str) -> list[str]:
    """Remove footers, bare page numbers, and the top running header from a page."""
    lines = text.split("\n")
    cleaned = []
    for i, line in enumerate(lines):
        if FOOTER_RE.search(line):
            continue
        if BARE_PAGENUM_RE.match(line) and i >= len(lines) - 3:
            continue  # bare page number near the bottom = footer residue
        cleaned.append(line)
    # Drop the first 1-2 non-empty lines if they look like a running header
    # (short, no paragraph marker, not a bullet) — these repeat every page.
    out, dropped = [], 0
    for line in cleaned:
        s = line.strip()
        if dropped < 2 and s and not PARA_RE.match(line) and not s.startswith("•") \
           and len(s) < 60 and fm_id.split()[-1] not in s:
            dropped += 1
            continue
        out.append(line)
    return out


def parse_fm(pdf_path: str, fm_id: str) -> list[dict]:
    chunks = []
    current = None          # the paragraph currently being assembled
    current_chapter = None
    current_section = None

    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            if BLANK_PAGE_RE.search(raw):
                continue
            for line in strip_page_furniture(raw, fm_id):
                s = line.strip()
                if not s:
                    continue

                m = PARA_RE.match(line)
                if m:
                    # flush the previous paragraph
                    if current:
                        chunks.append(current)
                    para_id = m.group(1)
                    chapter = para_id.split("-")[0]
                    current_chapter = chapter
                    current = {
                        "fm_id": fm_id,
                        "para_id": para_id,
                        "chapter": chapter,
                        "section": current_section,
                        "page_start": pageno,
                        "text": m.group(2).strip(),
                        "xrefs": [],
                    }
                elif CAPS_HEADER_RE.match(s) and not s.startswith("•"):
                    # Section header between paragraphs — update context,
                    # flush any open paragraph first.
                    if current:
                        chunks.append(current)
                        current = None
                    current_section = s.title()
                elif FIGURE_CAP_RE.match(s) or TEMPLATE_NOISE_RE.search(s):
                    # Entered a sample-order template / figure block. Close the
                    # current doctrine paragraph and stop absorbing template text.
                    if current:
                        chunks.append(current)
                        current = None
                elif current is not None:
                    # continuation line of the current paragraph (handles
                    # page-break splits automatically, since footers/headers
                    # were already stripped)
                    current["text"] += " " + s

    if current:
        chunks.append(current)

    # Post-process: collapse whitespace, capture cross-references
    for c in chunks:
        c["text"] = re.sub(r"\s+", " ", c["text"]).strip()
        c["xrefs"] = XREF_RE.findall(c["text"])
        c["char_len"] = len(c["text"])
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_path")
    ap.add_argument("out_path")
    ap.add_argument("--fm-id", required=True, help='e.g. "FM 5-0"')
    args = ap.parse_args()

    chunks = parse_fm(args.pdf_path, args.fm_id)
    with open(args.out_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Quick summary to stderr
    lens = [c["char_len"] for c in chunks]
    print(f"Parsed {len(chunks)} paragraphs from {args.fm_id}", file=sys.stderr)
    if lens:
        print(f"  char_len: min={min(lens)} med={sorted(lens)[len(lens)//2]} "
              f"max={max(lens)}", file=sys.stderr)
        chapters = sorted({c['chapter'] for c in chunks},
                          key=lambda x: (len(x), x))
        print(f"  chapters/appendices seen: {chapters}", file=sys.stderr)


if __name__ == "__main__":
    main()
