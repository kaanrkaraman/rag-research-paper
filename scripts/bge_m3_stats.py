"""Paired bootstrap significance tests for the BGE-M3 additions.

Runs four pre-registered comparisons on per-query Recall@5 lists, then
applies Bonferroni correction over the family. Output is appended to
`paper_neurips/bge_m3_results.md` as a third Markdown section.

Comparisons:
  1. BGE-M3 hybrid+BGE rerank   vs  OpenAI hybrid+Cohere rerank
       (headline open-vs-closed pipeline claim)
  2. BGE-M3 dense                vs  OpenAI dense (text-embed-3-large)
       (apples-to-apples encoder comparison)
  3. BGE-M3 hybrid+BGE rerank    vs  BGE-M3 hybrid (no rerank)
       (does the local reranker pay off?)
  4. BGE-M3 hybrid               vs  BM25
       (does adding open dense to BM25 still help?)

Reuses `paired_bootstrap_test` and `bonferroni_correction` from
`src/evaluation/statistical_tests.py`. No new statistics implemented.

Usage:
    python3 scripts/bge_m3_stats.py
    python3 scripts/bge_m3_stats.py --metric recall@10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.statistical_tests import (
    bonferroni_correction,
    paired_bootstrap_test,
)
from src.utils.common import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

RESULT_DIRS = [PROJECT_ROOT / "data" / "results", PROJECT_ROOT / "results"]


def _find_json(filename: str) -> Path | None:
    for d in RESULT_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def _load_per_query(filename: str, metric: str) -> list[float] | None:
    p = _find_json(filename)
    if p is None:
        return None
    with open(p) as f:
        d = json.load(f)
    pq = d.get("per_query_results", [])
    if not pq:
        return None
    # Sort by query_id so paired alignment matches across files.
    pq_sorted = sorted(pq, key=lambda q: q.get("query_id", q.get("id", "")))
    out = [q.get(metric) for q in pq_sorted]
    return out if all(v is not None for v in out) else None


COMPARISONS: list[tuple[str, str, str]] = [
    ("BGE-M3 hybrid+BGE rerank vs OpenAI hybrid+Cohere rerank",
     "hybrid+bge_bge_m3_whole_doc.json",
     "hybrid_rrf+cohere_rerank_whole_doc.json"),
    ("BGE-M3 dense vs OpenAI dense (text-embed-3-large)",
     "dense_bge_m3_whole_doc.json",
     "dense_openai_whole_doc.json"),
    ("BGE-M3 hybrid+BGE rerank vs BGE-M3 hybrid (no rerank)",
     "hybrid+bge_bge_m3_whole_doc.json",
     "hybrid_bge_m3_whole_doc.json"),
    ("BGE-M3 hybrid vs BM25",
     "hybrid_bge_m3_whole_doc.json",
     "bm25_openai_large_whole_doc.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="recall@5",
                    help="Per-query metric to test (recall@5, recall@10, mrr@3, ndcg@10, map)")
    ap.add_argument("--n-resamples", type=int, default=10_000)
    ap.add_argument("--out", default=None,
                    help="Append Markdown to this file (e.g. paper_neurips/bge_m3_results.md)")
    args = ap.parse_args()

    rows = []
    p_values = []
    for label, fa, fb in COMPARISONS:
        scores_a = _load_per_query(fa, args.metric)
        scores_b = _load_per_query(fb, args.metric)
        if scores_a is None or scores_b is None:
            rows.append({
                "label": label,
                "missing": [name for name, scores in [(fa, scores_a), (fb, scores_b)]
                            if scores is None],
            })
            p_values.append(None)
            continue
        if len(scores_a) != len(scores_b):
            rows.append({
                "label": label,
                "error": f"length mismatch ({len(scores_a)} vs {len(scores_b)})",
            })
            p_values.append(None)
            continue
        res = paired_bootstrap_test(scores_a, scores_b, n_resamples=args.n_resamples)
        rows.append({"label": label, **res, "n": len(scores_a)})
        p_values.append(res["p_value"])

    # Bonferroni over the family of valid comparisons.
    valid_p = [p for p in p_values if p is not None]
    if valid_p:
        adj = bonferroni_correction(valid_p)
        adj_iter = iter(adj)
        for r in rows:
            if "p_value" in r:
                r["bonferroni"] = next(adj_iter)

    # Render Markdown.
    out = ["## BGE-M3 paired bootstrap (per-query " + args.metric + ")\n"]
    out.append(f"_n_resamples = {args.n_resamples}, B-corrected over m = {len(valid_p)} comparisons._\n")
    out.append("| Comparison | n | mean diff | p (raw) | p < α (raw) | sig (Bonferroni) | 95% CI |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        if "missing" in r:
            out.append(f"| {r['label']} | — | — | — | — | — | MISSING: {', '.join(r['missing'])} |")
            continue
        if "error" in r:
            out.append(f"| {r['label']} | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        bonf = r.get("bonferroni", {})
        out.append(
            f"| {r['label']} | {r['n']} | {r['mean_diff']:+.4f} | "
            f"{r['p_value']:.4g} | {'yes' if r['significant_005'] else 'no'} | "
            f"{'yes' if bonf.get('significant') else 'no'} | "
            f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] |"
        )
    md = "\n".join(out) + "\n"
    print(md)

    if args.out:
        outpath = Path(args.out)
        existing = outpath.read_text() if outpath.exists() else ""
        outpath.write_text(existing + "\n" + md)
        logger.info(f"Appended to {outpath}")


if __name__ == "__main__":
    main()
