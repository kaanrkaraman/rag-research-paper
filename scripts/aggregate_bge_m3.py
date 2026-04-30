"""Aggregate BGE-M3 + baseline results into paper-paste Markdown.

Reads result JSONs from both `results/` (original OpenAI runs) and
`data/results/` (newer BGE-M3 runs), recomputes per-subset metrics from
`per_query_results`, and prints / saves Markdown tables that map directly
onto Tables 2, 5, 8 of `paper_neurips/main.tex`.

Run after `notebooks/bge_m3_h100.ipynb` has produced
`data/results/hybrid+bge_bge_m3_whole_doc.json`. Safe to run repeatedly:
the script is read-only on the repo state.

Usage:
    python3 scripts/aggregate_bge_m3.py
    python3 scripts/aggregate_bge_m3.py --out paper_neurips/bge_m3_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

# Where to look for result JSONs. Newer BGE-M3 runs land in `data/results/`;
# original OpenAI runs sit in `results/` at repo root.
RESULT_DIRS = [PROJECT_ROOT / "data" / "results", PROJECT_ROOT / "results"]


# Method label, file pattern (matched from the END of filename), display order.
# The order of this list determines table row order.
METHOD_PANEL: list[tuple[str, str]] = [
    ("BM25 (sparse)",                  "bm25_openai_large_whole_doc.json"),
    ("Dense (text-embed-3-large)",     "dense_openai_whole_doc.json"),
    ("Dense (BGE-M3)",                 "dense_bge_m3_whole_doc.json"),
    ("HyDE (gpt-4.1-mini)",            "hyde_gpt41mini_whole_doc.json"),
    ("Multi-Query + RRF",              "multi_query_gpt41mini_whole_doc.json"),
    ("Contextual Dense",               "contextual_dense_whole_doc.json"),
    ("Contextual Hybrid",              "contextual_hybrid_whole_doc.json"),
    ("CRAG (gpt-4.1-mini)",            "crag_whole_doc.json"),
    ("Hybrid (BM25+Dense, RRF)",       "hybrid_rrf_whole_doc.json"),
    ("Hybrid (BM25+BGE-M3, RRF)",      "hybrid_bge_m3_whole_doc.json"),
    ("Hybrid + Cohere Rerank",         "hybrid_rrf+cohere_rerank_whole_doc.json"),
    ("Hybrid (BGE-M3) + BGE Rerank",   "hybrid+bge_bge_m3_whole_doc.json"),
]


METRICS_FULL = [
    "recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
    "mrr@3", "mrr@5", "ndcg@5", "ndcg@10", "map",
]
METRICS_MAIN = ["recall@1", "recall@3", "recall@5", "recall@10",
                "mrr@3", "ndcg@10", "map"]
METRICS_SUBSET = ["recall@5", "recall@10", "mrr@3"]


def _find_json(filename: str) -> Path | None:
    for d in RESULT_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def _load_method(filename: str) -> dict | None:
    p = _find_json(filename)
    if p is None:
        return None
    with open(p) as f:
        return json.load(f)


def _per_subset_means(per_query: list[dict]) -> dict[str, dict[str, float]]:
    """Group per_query_results by `subset` and average each metric."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in per_query:
        s = q.get("subset", "?")
        for m in METRICS_FULL + ["recall@20"]:
            v = q.get(m)
            if v is not None:
                buckets[s][m].append(v)
    return {s: {m: sum(vs) / len(vs) for m, vs in mx.items() if vs}
            for s, mx in buckets.items()}


def _fmt(v: float | None, places: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{places}f}"


def main_table(loaded: list[tuple[str, dict | None]]) -> str:
    """Build a Markdown table matching paper_neurips/main.tex Table 2 columns."""
    header_cols = ["Method"] + METRICS_MAIN
    lines = ["| " + " | ".join(header_cols) + " |",
             "|" + "|".join(["---"] * len(header_cols)) + "|"]
    for label, data in loaded:
        if data is None:
            row = [label] + ["**MISSING**"] * len(METRICS_MAIN)
        else:
            m = data.get("retrieval_metrics", {})
            row = [label] + [_fmt(m.get(k)) for k in METRICS_MAIN]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def full_table(loaded: list[tuple[str, dict | None]]) -> str:
    """Markdown table matching Table 5 (full retrieval metrics, sortable by nDCG@10)."""
    header_cols = ["Method"] + METRICS_FULL
    rows: list[tuple[float, list[str]]] = []
    for label, data in loaded:
        if data is None:
            rows.append((-1.0, [label] + ["**MISSING**"] * len(METRICS_FULL)))
            continue
        m = data.get("retrieval_metrics", {})
        rows.append((
            m.get("ndcg@10", -1.0),
            [label] + [_fmt(m.get(k)) for k in METRICS_FULL],
        ))
    rows.sort(key=lambda r: r[0])
    out = ["| " + " | ".join(header_cols) + " |",
           "|" + "|".join(["---"] * len(header_cols)) + "|"]
    for _, r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def subset_table(loaded: list[tuple[str, dict | None]]) -> str:
    """Markdown table for Table 8 — per-subset R@5 / R@10 / MRR@3."""
    subsets = ["FinQA", "ConvFinQA", "TAT-DQA"]
    header = ["Subset", "Method", "N"] + METRICS_SUBSET
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for s in subsets:
        for label, data in loaded:
            if data is None:
                out.append(f"| {s} | {label} | — | " + " | ".join(["**MISSING**"] * 3) + " |")
                continue
            pq = data.get("per_query_results", [])
            if not pq:
                continue
            sub = [q for q in pq if q.get("subset") == s]
            if not sub:
                continue
            cells = [_fmt(sum(q.get(m, 0) for q in sub) / len(sub)) for m in METRICS_SUBSET]
            out.append(f"| {s} | {label} | {len(sub)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Optional path to save Markdown output (default: stdout only)")
    args = ap.parse_args()

    loaded: list[tuple[str, dict | None]] = []
    missing: list[str] = []
    for label, fname in METHOD_PANEL:
        data = _load_method(fname)
        loaded.append((label, data))
        if data is None:
            missing.append(f"{label}  ({fname})")

    sections: list[str] = []
    sections.append("# BGE-M3 results (paste-buffer)\n")
    sections.append(f"_Generated by `scripts/aggregate_bge_m3.py`. {len([d for _, d in loaded if d])} of "
                    f"{len(loaded)} method JSONs found._\n")

    if missing:
        sections.append("**Missing JSONs (skipped in tables below):**\n")
        for m in missing:
            sections.append(f"- {m}")
        sections.append("")

    sections.append("## Table 2 — main retrieval results\n")
    sections.append(main_table(loaded))
    sections.append("")

    sections.append("## Table 5 — full retrieval metrics (sorted by nDCG@10 ascending)\n")
    sections.append(full_table(loaded))
    sections.append("")

    sections.append("## Table 8 — per-subset breakdown\n")
    sections.append(subset_table(loaded))
    sections.append("")

    out = "\n".join(sections)
    print(out)

    if args.out:
        Path(args.out).write_text(out)
        logger.info(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
