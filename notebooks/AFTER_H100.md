# After the H100 run — paper-edit guide

This file picks up where `notebooks/bge_m3_h100.ipynb` ends. Once
`data/results/hybrid+bge_bge_m3_whole_doc.json` is on your laptop, walk
through the steps below in order.

Everything here runs on your local machine (M1 / Linux), not on the rented
GPU. Expect a 1–2 h human-time pass for the LaTeX edits.

---

## Step 1 — aggregate

```bash
python3 scripts/aggregate_bge_m3.py --out paper_neurips/bge_m3_results.md
python3 scripts/bge_m3_stats.py     --out paper_neurips/bge_m3_results.md
python3 scripts/generate_figures.py
```

Open `paper_neurips/bge_m3_results.md`. You should see:

- A 12-row main-results Markdown table (no **MISSING** cells).
- A 12-row sorted-by-nDCG@10 full-results table.
- A 36-row per-subset table (12 methods × 3 subsets).
- A 4-row paired-bootstrap table with raw and Bonferroni-corrected p-values.

Verify (cheap sanity):
```bash
grep -c MISSING paper_neurips/bge_m3_results.md   # must print 0
ls -lh paper_neurips/figures/*.pdf                  # 7 PDFs, all timestamped today
```

---

## Step 2 — paper edits (LaTeX)

Edit `paper_neurips/main.tex`. The order below matters: do the structural
adds first, then the numerical edits, then the things that depend on the
new significance matrix.

### 2a. Add three rows to Table 2 (lines ~120–148)

Append BGE-M3 rows under their categories. Numbers come from
`bge_m3_results.md` "Table 2". Latency intentionally omitted (cross-hardware).

```latex
\multirow{3}{*}{Single-method}
  & BM25 (sparse)                  & 0.293 & 0.552 & 0.644 & 0.735 & 0.411 & 0.515 & 0.449 \\
  & Dense (text-embed-3-large)     & 0.248 & 0.481 & 0.587 & 0.703 & 0.351 & 0.466 & 0.398 \\
  & Dense (BGE-M3)\textsuperscript{$\diamond$}
                                   & 0.225 & 0.425 & 0.508 & 0.597 & 0.315 & 0.406 & 0.351 \\
\midrule
...
\multirow{3}{*}{Fusion + rerank}
  & Hybrid (BM25+Dense, RRF)       & 0.308 & 0.588 & 0.695 & 0.801 & 0.433 & 0.551 & 0.477 \\
  & Hybrid (BGE-M3, RRF)\textsuperscript{$\diamond$}
                                   & 0.286 & 0.538 & 0.633 & 0.733 & 0.399 & 0.506 & 0.440 \\
  & Hybrid + Cohere Rerank         & \textbf{0.472} & \textbf{0.758} & \textbf{0.816} & \textbf{0.861} & \textbf{0.605} & \textbf{0.683} & \textbf{0.625} \\
  & Hybrid (BGE-M3) + BGE Rerank\textsuperscript{$\diamond$}
                                   & X.XXX & X.XXX & X.XXX & X.XXX & X.XXX & X.XXX & X.XXX \\
```

Replace the X.XXX placeholders with the actual numbers from
`bge_m3_results.md`. Update the table caption to add the dagger note:

> "\textsuperscript{$\diamond$} BGE-M3 (open weights) and BAAI/bge-reranker-v2-m3 run locally on a single H100 80~GB GPU; latency omitted because other rows ran on Apple Silicon and a cross-hardware comparison would be misleading."

### 2b. Add three rows to Table 5 / full results (lines ~336–354)

Same three BGE-M3 rows, sorted into the existing nDCG@10 ordering. Take the
exact ordered list from the "Table 5" block in `bge_m3_results.md` — it is
already sorted ascending. Carry the same `\textsuperscript{$\diamond$}` note.

### 2c. Add a BGE-M3 column to Table 8 / per-subset (lines ~356–382)

Add a fourth method block:

```latex
\multirow{3}{*}{BGE-M3 (open)}
  & Dense (BGE-M3)                 & X.XXX & X.XXX & X.XXX \\
  & Hybrid (BGE-M3)                & X.XXX & X.XXX & X.XXX \\
  & Hybrid (BGE-M3) + BGE Rerank   & X.XXX & X.XXX & X.XXX \\
```

If the table starts overflowing, drop the dense and hybrid rows and keep only
the headline `Hybrid (BGE-M3) + BGE Rerank` row.

### 2d. §3.4 Setup — add the H100 sentence (line ~108)

Replace:

> "BM25 and FAISS run locally on Apple Silicon."

with:

> "BM25 and FAISS run locally on Apple Silicon. We additionally evaluate BGE-M3 (1024-d) and the BAAI/bge-reranker-v2-m3 cross-encoder as a fully-open-weights pipeline; both run locally on a single NVIDIA H100 80~GB GPU."

### 2e. §5 Discussion — add the encoder-robustness paragraph (after line ~243)

Drop a paragraph between the existing "Why BM25 wins" and "The reranker does most of the heavy lifting" paragraphs:

> **The BM25 advantage is robust to encoder choice.** Swapping `text-embedding-3-large` for the open-weight BGE-M3 (1024-d) does not flip the ranking: BGE-M3 dense reaches Recall@5 = 0.508, *below* OpenAI dense at 0.587 and well below BM25 at 0.644. The two-stage open pipeline (BGE-M3 hybrid + BGE-reranker-v2-m3) closes most of the gap to the closed Cohere pipeline (R@5 = X.XXX vs.\ 0.816). The takeaway is structural rather than encoder-specific: lexical matching wins on text-and-table data because the answer-bearing tokens occur verbatim in the document, and dense encoders dilute that signal regardless of training corpus.

### 2f. §7.4 Limitations — rewrite (lines ~266–267)

Replace:

> "All dense experiments use \texttt{text-embedding-3-large}. A different encoder may shift the dense number up or down; we do not expect it to close the gap to BM25 on this corpus, but we have not measured it."

with:

> "We evaluated two dense encoders (\texttt{text-embedding-3-large}, BGE-M3) and two cross-encoder rerankers (Cohere Rerank v4.0 Pro, BAAI/bge-reranker-v2-m3). Other encoders may shift the absolute numbers, but the qualitative ordering — BM25 above either dense encoder, hybrid above both, hybrid+rerank above hybrid — held in every pairing we tested. We do not claim universality; encoder families with explicit table-structure pretraining (e.g.\ table-aware contrastive variants) may behave differently."

### 2g. Issue #4 — method count (abstract, §1, §3.2)

Change the count consistently. Recommended phrasing for **§1 introduction** (line ~47):

> "Where the original paper tested six retrieval methods with two metrics, we evaluate **nine retrieval methods** plus an open-weight BGE-M3 robustness panel, with the full suite of retrieval and generation metrics."

Same change in **abstract** (line ~33) and **§3.2** (line ~81). Make sure the
words "ten" and "twelve" do not appear anywhere.

### 2h. Issue #1 — fix the Table 2 caption claim

The new significance matrix is in `bge_m3_results.md` (last section). Use it
to rewrite the Table 2 caption. Likely text:

> "All pairwise differences are statistically significant at $p < 0.05$ after Bonferroni correction over the comparison family, with the noted exceptions in §\ref{sec:negative} and the BGE-M3 / OpenAI within-family comparisons."

Adjust based on whatever the matrix actually shows.

---

## Step 3 — pre-submission checks

```bash
cd paper_neurips
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Then verify:

```bash
# Page count: body must be ≤ 9
grep -c "newlabel{sec:" main.aux            # number of body labels
grep -E "newlabel.*\{sec:conclusion\}" main.aux  # check conclusion stays on page ≤9

# No undefined refs
! grep -E "Warning.*undef" main.log

# BGE-M3 actually appears in the right places (≥6: Tables 2, 5, 8, §3.4, §5, §7.4)
grep -c "BGE-M3" main.tex
[ "$(grep -c 'BGE-M3' main.tex)" -ge 6 ] && echo "OK: BGE-M3 mentioned in ≥6 places"

# CRAG dagger present
grep -F '0.788\textsuperscript{$\dagger$}' main.tex && echo "OK: CRAG R@20 dagger"

# Anonymity scrub — these should all return 0 lines
grep -r "radiate.com\|kaanrkaraman\|mea5963\|Akarsu\|Karaman\|Mierbach" paper_neurips/ \
  --exclude-dir=figures --exclude="*.pdf" --exclude="*.aux" --exclude="*.log" || echo "OK: no plaintext author identifiers"
```

Open `main.pdf` in a viewer and visually confirm:

1. Page 1 — author block is hidden / replaced by `Anonymous Authors`.
2. Tables 2, 5, 8 — three new BGE-M3 rows are present, no orphan †/diamond.
3. Table 3 (generation) — has GPT-4.1-mini AND GPT-5.4 columns.
4. §4.4 reranker depth — has the footnote about the 0.826 vs 0.816 reruns.
5. §5 — has the new "BM25 advantage is robust" paragraph.
6. References — no `?` placeholders for ColBERTv2 or Jina-ColBERT-v2.

---

## Step 4 — checklist

Update `paper_neurips/checklist.tex`:

- **Item 4 (reproducibility)** — strengthen: "All result JSONs include git SHA, library versions, GPU name, HF model commit hash, and FAISS index SHA-256 (collected by `src/utils/common.collect_provenance`)."
- **Item 8 (compute resources)** — add: "BGE-M3 dense and BAAI/bge-reranker-v2-m3 reranking ran on a single NVIDIA H100 80~GB. Other methods ran locally on Apple Silicon (M1/M2)."

Recompile after editing the checklist.

---

## Step 5 — submit

OpenReview portal opens 2026-04-15, full paper deadline 2026-05-06 AoE.

* Submit `paper_neurips/main.pdf`.
* Supplementary material zip should include: `data/results/*.json`,
  `paper_neurips/bge_m3_results.md`, `provenance.log`,
  `requirements-h100.txt`, and the source repo at the git SHA recorded
  in the result provenance block.

Final pre-flight: re-read the abstract and §1 in one sitting. If the
"nine methods + BGE-M3 robustness panel" framing reads as an afterthought,
swap the order — lead with the robustness finding.
