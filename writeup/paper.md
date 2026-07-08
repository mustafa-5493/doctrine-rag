# Doctrine-RAG: Quantifying the Quality–Latency–Memory Tradeoff of Retrieval-Augmented Generation on a Consumer GPU

A case study in deploying a full RAG pipeline over US Army doctrine on a 4GB GTX 1050, with an independent LLM judge and statistical validation of every claim.

---

## Abstract

We build a retrieval-augmented generation (RAG) system over a US Army Field Manual (FM 5-0, *Planning and Orders Production*) and use it to measure what running such a system on low-end consumer hardware actually costs in answer quality. A hybrid (lexical + dense) retriever is evaluated against a set of 299 human-verified questions; a quantized 3B generator (Qwen2.5-3B-Instruct) is benchmarked across quantization levels and GPU-offload configurations on a 4GB GTX 1050; and answer quality is scored by an independent, different-family LLM judge (Claude). Every quantitative claim is accompanied by bootstrap confidence intervals and paired significance tests, and the judge itself is validated for test-retest reliability.

Three findings stand out. (1) Hybrid retrieval consistently outperforms lexical-only and dense-only retrieval, and lexical vs. dense retrieval exhibit *opposite* sensitivities to false-premise question phrasing. (2) Partial GPU offload is *slower* than pure-CPU execution — a PCIe-transfer "valley" — so on constrained hardware the meaningful choice is full offload or none. (3) 8-bit quantization yields statistically significant improvements over 4-bit in faithfulness and citation accuracy but only a marginal (non-significant) correctness gain, at 54% higher VRAM cost and no speed benefit; on a 4GB card, 4-bit is the better practical default.

---

## 1. Motivation

Most RAG demonstrations run once in a cloud notebook on a powerful accelerator and report a single accuracy number. That tells you little about deployment on the hardware many real settings actually have: an old office laptop, a cheap GPU instance, or a machine a user already owns. This project asks a narrower, more practical question:

> **What is the real cost, in answer quality, of running a complete RAG pipeline on a low-end consumer GPU rather than a cloud accelerator — and where are that hardware's limits?**

We answer it with measured numbers rather than intuition, and we hold the evaluation to a standard high enough that the conclusions are defensible: confidence intervals, significance tests, and a validated judge.

## 2. Corpus and preprocessing

The corpus is **FM 5-0 (INCL C1, Nov 2024)**, a 412-page born-digital PDF. A paragraph-aware parser extracts the manual's natively numbered doctrine paragraphs (e.g. `1-137`, `G-3`) as retrieval units, preserving the citable paragraph identifier, chapter, and nearest section header. The parser stitches paragraphs across page boundaries, strips running headers and alternating footers, skips blank filler pages, and excludes the fill-in-the-blank order-format templates in the appendices (which are not doctrine prose). This yields **1,043 paragraphs** spanning all 8 chapters and 7 appendices, with a median length of 572 characters.

Preserving the doctrine-native paragraph numbering matters: it makes every retrieved unit citable the way a soldier or analyst would cite it, and it gives the evaluation a natural gold-label (see §4).

## 3. Retrieval

Paragraphs are embedded with **BAAI/bge-small-en-v1.5** (384-dim), L2-normalized so inner product equals cosine similarity, and indexed with an exact FAISS `IndexFlatIP`. A parallel **BM25** lexical index is built over the same paragraphs with a tokenizer that preserves hyphenated and dotted doctrine tokens (e.g. `METT-TC`, `FM 5-0`) rather than shattering them.

The hybrid retriever fuses the two rankings with **Reciprocal Rank Fusion (RRF)**. RRF fuses by rank rather than raw score, which sidesteps the fact that BM25 scores and cosine similarities live on incompatible scales. All retrieval runs on CPU; only generation uses the GPU.

## 4. Evaluation set

Retrieval and generation metrics require questions with known correct source paragraphs. We generate these *from* the corpus so the gold label comes for free: for a length-filtered, chapter-stratified sample of paragraphs, an LLM drafts (a) a natural "vanilla" question with a concise gold answer, and (b) a "negative" question containing a plausible false premise with a correcting answer. The paragraph a question was generated from *is* its gold retrieval target.

Candidate questions passed through an automated faithfulness screen (each answer checked against its own source paragraph, pinned to that paragraph as sole ground truth), and every flagged item plus a random sample of passing items was human-verified against source text. One malformed negative question was dropped. The final set is **150 vanilla + 149 negative = 299 verified questions**, each carrying its gold paragraph identifier.

## 5. Generation and hardware

The generator is **Qwen2.5-3B-Instruct**, run locally via `llama.cpp` (CUDA build) on an **NVIDIA GTX 1050 (4GB VRAM)**. We test two quantization levels from the official GGUF release — **Q4_K_M** (4-bit) and **Q8_0** (8-bit) — and, for each, sweep GPU-layer offload across `{0, 8, 16, 24, all}`. Retrieval runs on CPU; only the generator touches the GPU. Generation uses greedy decoding (temperature 0), a 4096-token context, and the top-5 retrieved paragraphs as context.

## 6. Judging

Answer quality is scored by **Claude (Anthropic API)** — deliberately a *different model family* from the Qwen generator, avoiding self-grading bias. For each answer the judge returns, against the human-verified gold answer and the retrieved passages: **correctness** (1–5, vs. gold facts), **faithfulness** (1–5, grounding in retrieved passages), and **citation accuracy** (whether the gold paragraph is cited). A fixed rubric is used for every condition.

Because greedy decoding is deterministic and offload only changes *where* computation happens, quality depends on quantization level, not offload configuration. We therefore judge one representative config per quant level and separately verify this assumption (§7.4).

---

## 7. Results

### 7.1 Hybrid retrieval wins, and lexical/dense diverge on false premises

On the 150 vanilla questions:

| Method | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.693 | 0.833 | 0.900 | 0.940 | 0.780 |
| Dense | 0.747 | 0.920 | 0.947 | 0.980 | 0.835 |
| **Hybrid** | **0.753** | **0.940** | **0.967** | 0.980 | **0.844** |

Hybrid is best or tied-best at every cutoff. Retrieval is not the system bottleneck: the correct paragraph appears in the top 10 for 98% of questions.

A second, less expected result emerges from the false-premise (negative) split. Lexical and dense retrieval have *opposite* sensitivities to false-premise phrasing: BM25 performs **better** on negative questions than vanilla ones (the false premise restates the paragraph's own terminology, giving BM25 more exact anchors), while dense retrieval performs slightly **worse** (the question's semantic content, by construction, contradicts its target passage). Hybrid retrieval is robust to this asymmetry, remaining strong on both question types. This is a concrete demonstration of *why* hybrid retrieval is the right architectural choice, beyond a marginal average-case gain.

*(Figure: `retrieval_comparison.png`)*

### 7.2 Partial GPU offload is slower than none

Sweeping GPU-offload on the GTX 1050 reveals a counterintuitive throughput curve. Adding GPU layers does **not** monotonically increase speed — throughput dips in the middle of the range before recovering:

| Quant | 0 layers | 8 | 16 | 24 | all | VRAM (all) |
|---|---|---|---|---|---|---|
| Q4_K_M | 6.69 | 4.51 | 6.70 | 8.69 | 12.62 | 2384 MB |
| Q8_0 | 5.27 | 3.67 | 3.08 | 7.59 | 11.66 | 3670 MB |

*(throughput in tokens/sec)*

Partial offload (8–16 layers) is often *slower than pure CPU execution*: every generated token must shuttle activations across the PCIe bus between CPU and GPU memory, and at low offload counts you pay that transfer cost without placing enough compute on the GPU to amortize it. Only near-full offload recovers and exceeds CPU throughput. The practical takeaway for constrained hardware is direct: **offload fully or not at all — the middle is the worst of both worlds.**

*(Figure: `offload_valley.png`)*

### 7.3 The 4GB memory ceiling

VRAM scales linearly with offloaded layers. At full offload, Q8_0 consumes **3670 MB — 90% of the 4GB card** — while Q4_K_M uses 2384 MB (58%). Q8 fits, but barely; a larger model, a longer context window, or additional KV-cache pressure would exceed the ceiling. This is the empirical boundary of this hardware class for a 3B-parameter generator.

### 7.4 8-bit vs 4-bit quality: significant for faithfulness and citation, marginal for correctness

Scoring the representative full-offload configs (n = 75 per quant), with bootstrap 95% CIs and paired tests (Wilcoxon for the 1–5 metrics, McNemar for binary citation):

| Metric | Q4_K_M | Q8_0 | Δ | Test | p | Verdict |
|---|---|---|---|---|---|---|
| Correctness | 4.16 [3.88, 4.41] | 4.35 [4.09, 4.57] | +0.19 | Wilcoxon | 0.063 | borderline |
| Faithfulness | 4.72 [4.53, 4.87] | 4.85 [4.71, 4.96] | +0.13 | Wilcoxon | 0.039 | **significant** |
| Citation acc. | 0.61 [0.51, 0.72] | 0.75 [0.64, 0.84] | +0.13 | McNemar | 0.031 | **significant** |

8-bit produces statistically significant improvements in faithfulness (p = 0.039) and citation accuracy (p = 0.031). The correctness improvement is **borderline** (p = 0.063) — likely a small true effect that n = 75 is underpowered to confirm, rather than noise (see §7.6). Citation accuracy is the weakest metric overall (61–75%): the correct paragraph is usually in context (R@5 ≈ 0.97), but the model does not always cite it explicitly — and this is the behavior most improved by the higher-precision quantization.

*(Figure: `quality_tradeoff.png`)*

### 7.5 Offload is quality-neutral despite token-level divergence

Greedy decoding was found *not* to be bit-identical across CPU and GPU execution: only 60% (Q4) and 73% (Q8) of answers were textually identical across offload configurations, as floating-point differences flipped individual tokens. This raised a question — does that divergence affect quality?

A Friedman test (repeated-measures, across the five offload configs within each quant level) finds **no significant quality difference across offload configurations for any metric** (all p > 0.13). The token-level changes are semantically inert: quality varies by < 0.06 across offload configs, versus a 0.13–0.19 gap between quant levels. **Layer offload is a pure speed/memory optimization, orthogonal to answer quality** — practitioners can trade offload for memory headroom without quality concern.

*(Figure: `offload_neutral.png`)*

### 7.6 The judge is highly reliable

Because every quality number depends on an LLM judge running at non-zero temperature, we measured its test-retest reliability by re-scoring 100 answers a second time:

| Metric | Exact | Within ±1 | Weighted κ / κ |
|---|---|---|---|
| Correctness | 96% | 100% | 0.984 |
| Faithfulness | 99% | 100% | 0.985 |
| Citation | 100% | — | 1.000 |

Agreement is near-perfect (quadratic-weighted κ ≈ 0.98; citation κ = 1.0). This matters for interpretation: the borderline correctness result (§7.4) is **not** an artifact of judge noise, since the judge is highly self-consistent on that metric — it more likely reflects a small true effect underpowered at this sample size.

---

## 8. Discussion and recommendation

The results compose into a clear deployment recommendation for this hardware class. On a 4GB consumer GPU running a 3B RAG generator:

- **Use 4-bit (Q4_K_M) as the default.** It matches 8-bit on correctness (the difference is not significant), sacrifices a small, measurable amount of faithfulness and citation fidelity, and uses 58% of VRAM versus 90% — leaving headroom for longer context or a larger corpus. It is also marginally *faster*.
- **Reserve 8-bit (Q8_0) for cases where citation fidelity is critical** and memory is not the binding constraint, accepting that it nearly saturates the card.
- **Offload fully or not at all.** Partial offload is the worst configuration for throughput, and offload level has no effect on quality — so choose it purely on the speed/memory frontier.
- **Prefer hybrid retrieval**, which is robust to the lexical/dense asymmetry that single-method retrieval exhibits on adversarially-phrased questions.

## 9. Limitations

This is a **single-model, single-corpus, single-GPU case study**, and the conclusions should be read as scoped to that setting rather than as general laws. Specific limitations:

- **One corpus, one manual.** Findings may differ on other doctrine, other domains, or larger corpora where an approximate (rather than exact) index becomes necessary.
- **One generator family and two quant levels.** We did not test other 3B models, larger models (which would not fit at 8-bit), or intermediate quantization schemes.
- **Synthetic-then-verified eval set.** Questions were LLM-generated and human-verified against source, not authored independently; the gold answers reflect the source paragraph, not external ground truth.
- **Judge validity vs. reliability.** The judge is shown to be highly *reliable* (self-consistent); its *validity* rests on scoring against human-verified gold answers, not on an independent proof of correctness.
- **Sample size.** The quality comparison uses n = 75 per config; the borderline correctness result would benefit from a larger sample.
- **Speed variance.** Throughput figures are single-run; they are not reported with error bars.

## 10. Reproducibility

The pipeline is organized as independently runnable stages, each producing a durable artifact:

```
parse (PDF → paragraphs) → embed + index (FAISS + BM25) → hybrid retriever
   → eval-set generation + faithfulness screen + human verification
   → quantized generation benchmark (GTX 1050)
   → independent LLM-judge scoring
   → statistical analysis (bootstrap CIs, paired tests, Friedman)
   → judge reliability + figures
```

All headline performance numbers are measured on the target consumer GPU; corpus construction, embedding, and judging run on cloud/CPU resources, with the split documented at each stage. Retrieval, statistics, reliability, and plotting require no GPU and no API. The evaluation set, result tables, statistics, and figures are included in the repository.

---

*Corpus: FM 5-0 (public, US Army Publishing Directorate). Generator: Qwen2.5-3B-Instruct (Apache-2.0). Embeddings: BGE-small-en-v1.5. Judge: Claude (Anthropic API). This project is an independent technical study and is not affiliated with or endorsed by the US Army.*
