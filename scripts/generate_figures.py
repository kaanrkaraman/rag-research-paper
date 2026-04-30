#!/usr/bin/env python3
"""Generate publication-quality figures for paper.tex using Matplotlib.

Outputs PDF + PNG to paper_neurips/figures/ (the NeurIPS submission's figure
directory). Loads result JSONs from BOTH `data/results/` (newer BGE-M3 runs
written by run_experiment.py) and `results/` (original OpenAI runs); the
former takes precedence when a filename collides.

Figure 4 (retrieval-generation correlation) is intentionally fixed to its
original four-point panel (Dense / BM25 / Hybrid RRF / Oracle, GPT-4.1-mini)
because BGE-M3 has no end-to-end generation runs in this submission cycle.
The caption in `paper_neurips/main.tex` reflects this scope.

Usage: python scripts/generate_figures.py
"""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Search order: prefer newer per-run output dir, fall back to legacy `results/`.
RESULTS_DIRS = [
    os.path.join(ROOT, "data", "results"),
    os.path.join(ROOT, "results"),
]
OUTDIR = os.path.join(ROOT, "paper_neurips", "figures")

# ── Methods (ascending by overall Recall@5) ───────────────────────
METHODS = [
    ("hyde_gpt41mini_whole_doc.json", "HyDE"),
    ("dense_bge_m3_whole_doc.json", "Dense (BGE-M3)"),
    ("dense_openai_whole_doc.json", "Dense"),
    ("contextual_dense_whole_doc.json", "Ctx Dense"),
    ("hybrid_bge_m3_whole_doc.json", "Hybrid (BGE-M3)"),
    ("multi_query_gpt41mini_whole_doc.json", "Multi-Query"),
    ("bm25_openai_large_whole_doc.json", "BM25"),
    ("crag_whole_doc.json", "CRAG"),
    ("hybrid_rrf_whole_doc.json", "Hybrid RRF"),
    ("contextual_hybrid_whole_doc.json", "Ctx Hybrid"),
    ("hybrid+bge_bge_m3_whole_doc.json", "Hybrid+BGE Rerank"),
    ("hybrid_rrf+cohere_rerank_whole_doc.json", "Hybrid+Rerank"),
]

# ── Visual encoding ──────────────────────────────────────────────
# BGE-M3 family colors are lighter relatives of their OpenAI/Cohere siblings,
# so figures still group "dense / hybrid / hybrid+rerank" visually.
COLOR = {
    "HyDE": "#d62728", "Dense": "#ff7f0e", "Ctx Dense": "#bcbd22",
    "Multi-Query": "#9467bd", "BM25": "#1f77b4", "CRAG": "#17becf",
    "Hybrid RRF": "#2ca02c", "Ctx Hybrid": "#8c564b", "Hybrid+Rerank": "#e377c2",
    "Dense (BGE-M3)": "#ffb87a",       # lighter orange (paired with Dense)
    "Hybrid (BGE-M3)": "#7fc97f",       # lighter green (paired with Hybrid RRF)
    "Hybrid+BGE Rerank": "#c45ca1",     # paired with Hybrid+Rerank
}
MARKER = {
    "HyDE": "v", "Dense": "s", "Ctx Dense": "D",
    "Multi-Query": "P", "BM25": "o", "CRAG": "^",
    "Hybrid RRF": "X", "Ctx Hybrid": "*", "Hybrid+Rerank": "h",
    "Dense (BGE-M3)": "<", "Hybrid (BGE-M3)": ">", "Hybrid+BGE Rerank": "p",
}
DASH = {
    "HyDE": (0, (3, 2)), "Dense": (0, (3, 2)), "BM25": (0, (3, 2)),
    "Ctx Dense": (0, (5, 2, 1, 2)), "Multi-Query": (0, (5, 2, 1, 2)), "CRAG": (0, (5, 2, 1, 2)),
    "Hybrid RRF": "solid", "Ctx Hybrid": "solid", "Hybrid+Rerank": "solid",
    "Dense (BGE-M3)": (0, (3, 2)), "Hybrid (BGE-M3)": "solid",
    "Hybrid+BGE Rerank": "solid",
}

# ── Layout constants (ACL two-column format) ────────────────────
COL_W = 3.33   # single-column width in inches
TEXT_W = 6.88   # full text width in inches (figure*)

# ── Matplotlib global style ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
})

METRIC_GREEN = "#2ca02c"
METRIC_BLUE = "#1f77b4"


# ── Helpers ──────────────────────────────────────────────────────
def load_json(fname):
    """Load a result JSON, searching `RESULTS_DIRS` in order."""
    for d in RESULTS_DIRS:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"{fname} not found in any of: {RESULTS_DIRS}"
    )


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  {name}.pdf + .png")


# ── Data loading ─────────────────────────────────────────────────
def load_all_data():
    """Load every method in METHODS; skip (with a warning) any that are missing.

    Mutates the module-level METHODS list to drop missing entries so all
    downstream figures iterate only over methods that actually have data.
    """
    global METHODS
    agg = {}
    sub_r5 = {}
    kept: list[tuple[str, str]] = []
    for fname, name in METHODS:
        print(f"  Loading {name}...")
        try:
            data = load_json(fname)
        except FileNotFoundError as e:
            print(f"    SKIP {name}: {e}")
            continue
        agg[name] = data["retrieval_metrics"]
        by_sub = defaultdict(list)
        for q in data["per_query_results"]:
            by_sub[q["subset"]].append(q["recall@5"])
        sub_r5[name] = {s: float(np.mean(v)) for s, v in by_sub.items()}
        kept.append((fname, name))
        del data
    METHODS = kept
    return agg, sub_r5


# ── Figure 1: Recall@k Curves (full-width figure*) ──────────────
def fig1_recall_at_k(agg):
    print("Figure 1: Recall@k curves")
    ks = [1, 3, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.8))

    for _, name in METHODS:
        rm = agg[name]
        avail = [k for k in ks if f"recall@{k}" in rm]
        vals = [rm[f"recall@{k}"] for k in avail]
        ax.plot(
            avail, vals, label=name,
            color=COLOR[name], marker=MARKER[name],
            linestyle=DASH[name], linewidth=1.6, markersize=6,
            markeredgecolor="white", markeredgewidth=0.6,
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0.15, 0.95)
    ax.legend(
        loc="lower right", ncol=3, framealpha=0.95,
        edgecolor="#cccccc", fancybox=False, handlelength=2.2,
        columnspacing=1.0, handletextpad=0.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "recall_at_k")


# ── Figure 2: Main Comparison Bar Chart (full-width figure*) ─────
def fig2_main_comparison(agg):
    print("Figure 2: Main comparison bars")
    metrics = [
        ("recall@5", "R@5"), ("recall@10", "R@10"),
        ("mrr@3", "MRR@3"), ("ndcg@10", "nDCG@10"), ("map", "MAP"),
    ]
    n_methods = len(METHODS)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    total_width = 0.82
    bar_w = total_width / n_methods

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.2))

    for i, (_, name) in enumerate(METHODS):
        rm = agg[name]
        vals = [rm[key] for key, _ in metrics]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        ax.bar(
            x + offset, vals, bar_w * 0.92,
            label=name, color=COLOR[name],
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.95)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15),
        ncol=5, framealpha=0.95, edgecolor="#cccccc",
        fancybox=False, fontsize=7.5, handlelength=1.2,
        columnspacing=0.8, handletextpad=0.4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "main_comparison")


# ── Figure 3: Subset Heatmap (single-column) ────────────────────
def fig3_subset_heatmap(sub_r5):
    print("Figure 3: Subset heatmap")
    subsets = ["ConvFinQA", "FinQA", "TAT-DQA"]
    names = [name for _, name in METHODS]
    z = np.array([[sub_r5[name][s] for s in subsets] for name in names])

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    im = ax.imshow(z, cmap="YlOrRd", aspect="auto", vmin=0.45, vmax=0.90)

    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels(subsets, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.xaxis.set_ticks_position("bottom")

    # Annotate cells
    for i in range(len(names)):
        for j in range(len(subsets)):
            val = z[i, j]
            color = "white" if val > 0.78 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    save(fig, "subset_heatmap")


# ── Figure 4: Retrieval-Generation Correlation (single-column) ──
def fig4_correlation(agg):
    print("Figure 4: Retrieval-generation correlation")
    gen = {g["tag"]: g["nm"] for g in load_json("generation_all_fixed.json")}

    points = [
        ("Dense",      "dense_gpt41mini",  "#ff7f0e", "s"),
        ("BM25",       "bm25_gpt41mini",   "#1f77b4", "o"),
        ("Hybrid RRF", "hybrid_gpt41mini", "#2ca02c", "X"),
        ("Oracle",     "oracle_gpt41mini",  "#7B3294", "D"),
    ]
    xs, ys = [], []
    for display, tag, _, _ in points:
        r5 = 1.0 if "oracle" in tag else agg[display]["recall@5"]
        xs.append(r5)
        ys.append(gen[tag])

    xs_a, ys_a = np.array(xs), np.array(ys)
    slope, intercept = np.polyfit(xs_a, ys_a, 1)
    r = float(np.corrcoef(xs_a, ys_a)[0, 1])
    xl = np.linspace(0.5, 1.08, 100)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(xl, slope * xl + intercept, "--", color="gray", alpha=0.5, linewidth=1.2)

    for i, (display, _, clr, mkr) in enumerate(points):
        ax.scatter(xs[i], ys[i], s=80, color=clr, marker=mkr,
                   edgecolors="white", linewidth=0.8, zorder=5,
                   label=display)

    ax.legend(loc="lower right", framealpha=0.95, edgecolor="#cccccc",
              fancybox=False, handletextpad=0.4)
    ax.text(0.03, 0.97, f"$r$ = {r:.3f}", transform=ax.transAxes,
            fontsize=9, va="top", ha="left", style="italic")
    ax.set_xlabel("Retrieval Recall@5")
    ax.set_ylabel("Generation Number Match")
    ax.set_xlim(0.52, 1.08)
    ax.set_ylim(0.23, 0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "retrieval_generation_correlation")


# ── Figure 5: Fusion Ablation (two-panel, full-width figure*) ───
def fig5_fusion_ablation():
    print("Figure 5: Fusion ablation")
    cc_alpha = [0.3, 0.5, 0.7, 0.9]
    cc_r5  = [0.703, 0.726, 0.698, 0.614]
    cc_mrr = [0.456, 0.466, 0.452, 0.401]

    rrf_k   = [10, 30, 60, 100]
    rrf_r5  = [0.716, 0.705, 0.695, 0.695]
    rrf_mrr = [0.435, 0.433, 0.433, 0.433]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.2), sharey=True)

    mkr = dict(markersize=6, markeredgecolor="white", markeredgewidth=0.6)

    # Top: Convex Combination
    ax1.plot(cc_alpha, cc_r5, "o-", color=METRIC_GREEN, linewidth=1.8, label="R@5", **mkr)
    ax1.plot(cc_alpha, cc_mrr, "s--", color=METRIC_BLUE, linewidth=1.8, label="MRR@3", **mkr)
    ax1.set_xlabel(r"$\alpha$ (dense weight)")
    ax1.set_ylabel("Score")
    ax1.set_title(r"Convex Combination: Effect of $\alpha$", fontsize=9)
    ax1.set_ylim(0.38, 0.76)
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom: RRF
    ax2.plot(rrf_k, rrf_r5, "o-", color=METRIC_GREEN, linewidth=1.8, label="R@5", **mkr)
    ax2.plot(rrf_k, rrf_mrr, "s--", color=METRIC_BLUE, linewidth=1.8, label="MRR@3", **mkr)
    ax2.set_xlabel("k (RRF smoothing)")
    ax2.set_ylabel("Score")
    ax2.set_title("RRF: Effect of k Parameter", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    save(fig, "fusion_ablation")


# ── Figure 6: Reranker Depth Ablation (single-column) ───────────
def fig6_reranker_depth():
    print("Figure 6: Reranker depth")
    raw = load_json("reranker_depth_ablation.json")
    order = [(20, 10), (50, 5), (50, 10), (50, 20), (100, 10)]
    lookup = {(d["candidates"], d["top_n"]): d["metrics"] for d in raw}

    labels = [f"{c}\u2192{n}" for c, n in order]
    r5  = [lookup[k]["recall@5"] for k in order]
    mrr = [lookup[k]["mrr@3"] for k in order]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.bar(x - w/2, r5, w, label="R@5", color=METRIC_GREEN, edgecolor="white", linewidth=0.4)
    ax.bar(x + w/2, mrr, w, label="MRR@3", color=METRIC_BLUE, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Candidates \u2192 Top-N")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "reranker_depth")


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    agg, sub_r5 = load_all_data()
    print("\nGenerating figures...")
    fig1_recall_at_k(agg)
    fig2_main_comparison(agg)
    fig3_subset_heatmap(sub_r5)
    fig4_correlation(agg)
    fig5_fusion_ablation()
    fig6_reranker_depth()
    print(f"\nDone. Figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
