# After the H100 / Blackwell rerun — paper update guide

Use this guide **after** the apple-to-apple rerun on Colab finishes (the one
launched with `bash scripts/run_h100_experiment.sh --skip-install --skip-sanity --force`
following the `max_seq_length=8192` fix in commit
`1493b03 — Fix BGE-M3 fair-comparison bug + paper edits with current numbers`).

The structural paper edits (Tables 2/5/8 layout, §3.2 paragraph, §3.4 Setup
disclosure, §5 robustness paragraph, Limitations rewrite, method-count
alignment, appendix Table 7 hyperparameter rows, Table 2 caption fix) have
**already been applied** with the *prior, max\_seq=1024* numbers. After the
rerun, only the BGE-M3 cells in three tables and a handful of derived
deltas in the §5 paragraph need to swap.

---

## Step 1 — sync results back

Pick one. Whichever you used to send F.1/F.2 over.

```bash
# Option A: rsync from the Colab runtime / mounted Drive, on your laptop
rsync -av <colab-host-or-drive>:/content/RAGPaper/data/results/ ./data/results/
rsync -av <colab-host-or-drive>:/content/RAGPaper/provenance.log ./

# Option B: download the three JSONs by hand
#   data/results/dense_bge_m3_whole_doc.json
#   data/results/hybrid_bge_m3_whole_doc.json
#   data/results/hybrid+bge_bge_m3_whole_doc.json
```

Verify each JSON's `config` block embeds the apple-to-apple operating-point
values:

```bash
python3 - <<'PY'
import json
for f in ("dense_bge_m3", "hybrid_bge_m3", "hybrid+bge_bge_m3"):
    d = json.load(open(f"data/results/{f}_whole_doc.json"))
    cfg = d["config"]
    print(f, "max_seq:", cfg.get("embedder_max_seq_length"),
          "rerank_max:", cfg.get("reranker_max_length"),
          "pool:", cfg.get("candidate_pool_size"),
          "records:", len(d["per_query_results"]))
PY
```

Expected output:

```
dense_bge_m3 max_seq: 8192 rerank_max: None pool: 60 records: 23088
hybrid_bge_m3 max_seq: 8192 rerank_max: None pool: 60 records: 23088
hybrid+bge_bge_m3 max_seq: 8192 rerank_max: 512 pool: 60 records: 23088
```

If `max_seq` is not 8192, the runtime did not pick up the new code — re-pull
on Colab and retry.

---

## Step 2 — regenerate the paste-buffer Markdown + figures + stats

```bash
python3 scripts/aggregate_bge_m3.py --out paper_neurips/bge_m3_results.md
python3 scripts/bge_m3_stats.py     --out paper_neurips/bge_m3_results.md
python3 scripts/generate_figures.py
```

Open `paper_neurips/bge_m3_results.md`. Verify:

- 12-row main table (no `**MISSING**` cells).
- 12-row sorted-by-nDCG@10 full-results table.
- 36-row per-subset table (12 methods × 3 subsets).
- 4-row paired-bootstrap table with raw and Bonferroni p-values.

```bash
grep -c MISSING paper_neurips/bge_m3_results.md   # must print 0
ls -lh paper_neurips/figures/*.pdf                  # 7 PDFs, today's timestamps
```

---

## Step 3 — number-only swaps in `paper_neurips/main.tex`

The structural edits are already in place. The list below is what changes
when the rerun completes. Each entry shows the **prior placeholder values**
that are currently in the file (post commit `1493b03`) and the **target**
(read from the new JSONs / `bge_m3_results.md`).

### 3a. Table 2 — three BGE-M3 rows (lines around `\begin{table*}` for `tab:main_results`)

Three rows currently carry placeholder F.1/F.2/F.3 numbers. Replace each
cell with the corresponding value from `bge_m3_results.md` "Table 2".
Specifically:

| Row | Cells to update (R@1, R@3, R@5, R@10, MRR@3, nDCG@10, MAP) | Currently |
|---|---|---|
| `Dense (BGE-M3)\textsuperscript{$\diamond$}` | 7 | 0.225 / 0.425 / 0.508 / 0.597 / 0.315 / 0.406 / 0.351 |
| `Hybrid (BGE-M3, RRF)\textsuperscript{$\diamond$}` | 7 | 0.286 / 0.538 / 0.633 / 0.733 / 0.399 / 0.507 / 0.440 |
| `Hybrid (BGE-M3) + BGE Rerank\textsuperscript{$\diamond$}` | 7 | 0.301 / 0.537 / 0.611 / 0.693 / 0.409 / 0.499 / 0.442 |

### 3b. Table 5 — three BGE-M3 rows in the sorted full-results block

Same three method rows. After the rerun, the **sort position** of each row
might shift up the table (if BGE-M3 nDCG@10 increases, e.g.\ moves past
HyDE / Multi-Query / BM25). Use the `nDCG@10` column from
`bge_m3_results.md` "Table 5 (sorted)" to decide where each row lives.

### 3c. Table 8 — per-subset rows (5 rows per subset block)

Two BGE-M3 rows per subset (Dense BGE-M3, Hybrid BGE-M3 + BGE Rerank);
six cells total per subset. Currently:

| Subset | Dense (BGE-M3) R@5 / R@10 / MRR@3 | Hybrid+BGE Rerank R@5 / R@10 / MRR@3 |
|---|---|---|
| FinQA      | 0.600 / 0.703 / 0.365 | 0.660 / 0.754 / 0.393 |
| ConvFinQA  | 0.582 / 0.677 / 0.330 | 0.687 / 0.759 / 0.492 |
| TAT-DQA    | 0.418 / 0.496 / 0.274 | 0.551 / 0.628 / 0.395 |

### 3d. §5 Discussion paragraph (the "BM25 advantage is robust" paragraph)

Five derived numbers in this paragraph:

1. `BGE-M3 dense reaches Recall@5 = 0.508` — replace with new BGE-M3 dense R@5.
2. `R@5 = 0.633 vs.\ 0.644, $-1.1$~pp` — replace 0.633 and recompute the delta vs.\ BM25's 0.644.
3. `R@5 = 0.611` (in `Hybrid (BGE-M3) + BGE Rerank, R@5 = 0.611`) — replace with new F.3 R@5.
4. `75\% of its top-20 ceiling` — recompute as `new_F3_R@5 / new_F2_R@20 * 100`.
5. The Cohere `93%` figure does **not** change (it's `0.816 / 0.877`, both closed-stack constants).

The `+5.1`~pp claim (OAI hybrid +5.1 over BM25) does not change either — both
operands are closed-stack.

### 3e. Table 2 caption

Currently rewrites the overstated "p<0.001 between adjacent methods" claim
with "p<0.05 after Bonferroni correction over the comparison family, with
the exceptions noted in §\ref{sec:negative}". After the rerun, look at the
paired-bootstrap table in `bge_m3_results.md` — if any *adjacent-row*
within-stack pairs are non-significant, name them in the caption. (The
existing exception phrasing already covers the OpenAI/BGE-M3 *between-stack*
near-ties under significance correction.)

### 3f. (Optional) Add a paragraph after Table 2 quantifying the BGE-M3 lift

Currently the post-table summary on line ~158 mentions only the closed-stack
numbers ("0.816 / 0.695 / 0.644 / 0.587"). After the rerun, consider adding
one sentence: "The open-weight stack reaches Recall@5 = X.XXX (Hybrid (BGE-M3)
+ BGE Rerank), Y~pp behind the closed-stack 0.816." Skip if the gap reads
worse than the current §5 framing supports.

---

## Step 4 — recompile + pre-submission checks

```bash
cd paper_neurips
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Verification:

```bash
# Body must end at page ≤9 (Conclusion currently lands on page 8).
grep "newlabel{sec:conclusion" main.aux

# No undefined refs.
grep -E "Warning.*undef" main.log

# BGE-M3 mentioned in ≥6 places (currently 24 — well above threshold).
grep -c "BGE-M3" main.tex

# CRAG dagger present.
grep -F '0.788\textsuperscript{$\dagger$}' main.tex

# Anonymity scrub — must return 0 hits.
grep -r "radiate.com\|kaanrkaraman\|mea5963\|Akarsu\|Karaman\|Mierbach" paper_neurips/ \
  --exclude-dir=figures --exclude="*.pdf" --exclude="*.aux" --exclude="*.log"
```

Open `main.pdf` and visually confirm:

1. Page 1 — `Anonymous Author(s) / Affiliation / Address / email` (auto-handled by `[eandd]`).
2. Tables 2, 5, 8 — three BGE-M3 rows present, every $\diamond$ has a matching footnote in the caption.
3. Table 3 (generation) — has both `GPT-4.1-mini` and `GPT-5.4` columns.
4. §4.4 (Reranker depth) — has the footnote about the 0.826 vs.\ 0.816 reruns.
5. §5 — has the new "BM25 advantage robust to encoder choice — and dense fusion is not" paragraph.
6. References — no `?` placeholders for `ColBERTv2` or `Jina-ColBERT-v2`.

---

## Step 5 — submit

NeurIPS 2026 E&D Track full-paper deadline: **2026-05-06 AoE**. OpenReview portal opens 2026-04-15.

* Submit `paper_neurips/main.pdf`.
* Supplementary zip: `data/results/*.json`, `paper_neurips/bge_m3_results.md`,
  `provenance.log`, `requirements-h100.txt`, and the source repo at the git SHA
  recorded in the per-result provenance block (see Step 1 verifier above).

Final pre-flight: re-read the abstract and §1 in one sitting. If the
"nine methods + open-weight BGE-M3 robustness panel" framing reads as an
afterthought, swap the order — lead with the robustness finding.
