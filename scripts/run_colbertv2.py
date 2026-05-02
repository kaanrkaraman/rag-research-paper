"""ColBERTv2 retrieval over the T2-RAGBench whole-document corpus.

Late-interaction retriever, the natural fix named in the paper's conclusion
for the 73% table-structure-mismatch failure mode. Runs as a fifth experiment
in the H100 bootstrap pipeline.

Tries ragatouille first (friendlier API, handles PLAID indexing automatically).
Falls back to bare colbert-ai if ragatouille's import path explodes against
the existing PyTorch / transformers versions on the H100 instance.

Output: data/results/colbertv2_whole_doc.json with the same schema as
scripts/run_experiment.py — config block, retrieval_metrics, per_query_results,
and a provenance sub-block including git SHA, library versions, GPU name, and
the index size on disk.

Document handling: ColBERTv2 was trained at max passage length 512. T2-RAGBench
docs average ~920 tokens, so we let ragatouille split each doc into 512-token
passages, then aggregate retrieval back to document level by max passage score
per (query, doc_id). This is the standard whole-document-from-passages pattern
documented in the ColBERT paper.

Usage:
    python3 scripts/run_colbertv2.py --top-k 20
    python3 scripts/run_colbertv2.py --top-k 20 --max-queries 100  # smoke test
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.data_loader import load_t2ragbench
from src.evaluation.retrieval_metrics import (
    compute_per_query_retrieval,
    compute_retrieval_metrics,
)
from src.utils.common import (
    RESULTS_DIR,
    ExperimentResult,
    RetrievedDoc,
    Timer,
    collect_provenance,
    get_logger,
)

logger = get_logger(__name__)


def _dir_size_files(path: Path) -> tuple[float, int]:
    """Return (size_in_MB, file_count) for a directory tree, 0/0 if missing."""
    if not path.exists():
        return 0.0, 0
    total = 0
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total / (1024 * 1024), n


def _gpu_status() -> str:
    """One-line GPU utilization summary; "n/a" if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip().split(",")
        util, used, total = (s.strip() for s in out)
        return f"gpu={util}% mem={used}/{total}MB"
    except Exception:
        return "gpu=n/a"


def _heartbeat(stop_event: threading.Event, watch_dir: Path,
               label: str, interval_s: int = 30) -> None:
    """Print a one-line liveness signal every interval_s while stop_event is unset.

    The indexing phase is otherwise silent for tens of minutes (ragatouille
    builds PLAID centroids and writes shards internally). This thread surfaces
    GPU activity + on-disk index growth so the user can tell whether progress
    is still being made.
    """
    t0 = time.time()
    while not stop_event.wait(interval_s):
        elapsed = int(time.time() - t0)
        size_mb, files = _dir_size_files(watch_dir)
        print(
            f"[heartbeat {label}] elapsed={elapsed//60:>3d}m{elapsed%60:02d}s  "
            f"{_gpu_status()}  index={size_mb:>7.1f}MB ({files} files)",
            flush=True,
        )

INDEX_ROOT = Path(__file__).parent.parent / "data" / "indexes" / "colbertv2"
CHECKPOINT = "colbert-ir/colbertv2.0"
INDEX_NAME = "t2ragbench_whole_doc"


def _load_ragatouille():
    """Import ragatouille; raise a clear error if it isn't installed."""
    try:
        from ragatouille import RAGPretrainedModel
        return RAGPretrainedModel
    except ImportError as e:
        raise SystemExit(
            "ragatouille is not installed. On the H100 bootstrap, the install "
            "is wired into scripts/run_h100_experiment.sh. To install manually:\n"
            "    pip install ragatouille\n"
            f"Underlying import error: {e}"
        )


def _passages_to_doc_ranking(passage_results: list[dict], top_k: int) -> list[RetrievedDoc]:
    """Aggregate ragatouille passage hits up to document level.

    ragatouille returns one entry per matched passage with the originating
    document_id. Group by document_id, take the max score per doc, sort by
    that max score, and return the top-k unique docs as RetrievedDoc objects.
    """
    best_per_doc: dict[str, dict] = {}
    for p in passage_results:
        doc_id = p.get("document_id") or p.get("doc_id") or p.get("id")
        if doc_id is None:
            continue
        score = float(p.get("score", 0.0))
        existing = best_per_doc.get(doc_id)
        if existing is None or score > existing["score"]:
            best_per_doc[doc_id] = {
                "doc_id": doc_id,
                "score": score,
                "text": p.get("content", "") or p.get("passage", ""),
            }
    ranked = sorted(best_per_doc.values(), key=lambda d: -d["score"])[:top_k]
    return [
        RetrievedDoc(
            doc_id=d["doc_id"],
            text=d["text"],
            score=d["score"],
            rank=i + 1,
            method="colbertv2",
        )
        for i, d in enumerate(ranked)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of documents to return per query (post-aggregation)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="Cap query count for sanity runs; full eval omits this")
    ap.add_argument("--max-document-length", type=int, default=512,
                    help="Passage chunk size in tokens (ColBERTv2 trained at 512)")
    ap.add_argument("--passage-oversample", type=int, default=10,
                    help="Retrieve top_k * oversample passages then dedupe to top_k docs")
    ap.add_argument("--bsize", type=int, default=64,
                    help="Indexer batch size on GPU")
    ap.add_argument("--output-name", default="colbertv2_whole_doc",
                    help="Output JSON name (without .json)")
    args = ap.parse_args()

    RAGPretrainedModel = _load_ragatouille()

    # ---------- Data ----------
    logger.info("Loading T2-RAGBench...")
    data = load_t2ragbench()
    qa_items = data.qa_items[:args.max_queries] if args.max_queries else data.qa_items
    docs = list(data.corpus.values())
    doc_ids = [d.doc_id for d in docs]
    doc_texts = [d.text for d in docs]
    logger.info(f"Corpus: {len(doc_ids)} docs, queries: {len(qa_items)}")

    # ---------- Index ----------
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    index_dir = INDEX_ROOT / INDEX_NAME

    index_timer = Timer()
    if index_dir.exists() and any(index_dir.rglob("*.pt")):
        logger.info(f"Loading cached ColBERTv2 index at {index_dir}")
        rag = RAGPretrainedModel.from_index(str(index_dir))
    else:
        print(f"[colbertv2] Building index from {CHECKPOINT} over "
              f"{len(doc_texts)} documents (passage_len={args.max_document_length}, "
              f"bsize={args.bsize}). This phase is normally silent for tens of "
              f"minutes; the heartbeat thread below prints liveness every 30s.",
              flush=True)
        stop_hb = threading.Event()
        hb = threading.Thread(
            target=_heartbeat,
            args=(stop_hb, index_dir, "indexing", 30),
            daemon=True,
        )
        hb.start()
        try:
            with index_timer:
                rag = RAGPretrainedModel.from_pretrained(CHECKPOINT, index_root=str(INDEX_ROOT))
                rag.index(
                    collection=doc_texts,
                    document_ids=doc_ids,
                    index_name=INDEX_NAME,
                    max_document_length=args.max_document_length,
                    split_documents=True,
                    bsize=args.bsize,
                )
        finally:
            stop_hb.set()
            hb.join(timeout=5)
        size_mb, n_files = _dir_size_files(index_dir)
        print(f"[colbertv2] Indexing finished in {index_timer.elapsed:.1f}s "
              f"({size_mb:.1f}MB across {n_files} files)", flush=True)

    # ---------- Retrieval ----------
    passages_per_query = max(args.top_k * args.passage_oversample, 50)
    all_retrieved: list[list[RetrievedDoc]] = []
    all_relevant_ids: list[set[str]] = []
    per_query: list[dict] = []
    latencies_ms: list[float] = []

    print(f"[colbertv2] Retrieving top-{args.top_k} docs per query "
          f"(oversample {passages_per_query} passages over {len(qa_items)} queries)",
          flush=True)
    pbar = tqdm(qa_items, desc="ColBERTv2 retrieval", mininterval=2.0,
                miniters=10, dynamic_ncols=True)
    for qa in pbar:
        t0 = time.perf_counter()
        passage_hits = rag.search(query=qa.question, k=passages_per_query)
        retrieved = _passages_to_doc_ranking(passage_hits, args.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        all_retrieved.append(retrieved)
        relevant = {qa.context_id}
        all_relevant_ids.append(relevant)

        pq_metrics = compute_per_query_retrieval(retrieved, relevant)
        per_query.append({
            "query_id": qa.id,
            "subset": qa.subset,
            "retrieved_ids": [r.doc_id for r in retrieved],
            "latency_ms": latency_ms,
            **pq_metrics,
        })
        latencies_ms.append(latency_ms)

        # Lightweight running R@5 in the bar so the user sees quality drift,
        # not just throughput. Updated every 100 queries to keep cost trivial.
        if len(per_query) % 100 == 0:
            running_r5 = sum(r.get("recall@5", 0.0) for r in per_query) / len(per_query)
            pbar.set_postfix(R5=f"{running_r5:.3f}", lat_ms=f"{latency_ms:.0f}")
    pbar.close()

    # ---------- Aggregate metrics ----------
    retrieval_metrics = compute_retrieval_metrics(all_retrieved, all_relevant_ids)
    logger.info(
        "ColBERTv2 retrieval metrics: "
        f"R@5={retrieval_metrics.get('recall@5'):.4f}, "
        f"R@10={retrieval_metrics.get('recall@10'):.4f}, "
        f"MRR@3={retrieval_metrics.get('mrr@3'):.4f}, "
        f"nDCG@10={retrieval_metrics.get('ndcg@10'):.4f}"
    )

    # ---------- Provenance ----------
    provenance = collect_provenance(
        embedding_model=CHECKPOINT,
        index_path=index_dir,
    )

    index_size_mb = 0.0
    if index_dir.exists():
        try:
            index_size_mb = sum(
                p.stat().st_size for p in index_dir.rglob("*") if p.is_file()
            ) / (1024 * 1024)
        except OSError:
            pass

    # ---------- Save ----------
    result = ExperimentResult(
        method="colbertv2",
        config={
            "method": "colbertv2",
            "embedding": "colbertv2",
            "reranker": "none",
            "top_k": args.top_k,
            "chunking": "passages_512_max_aggregated",
            "subset": "all",
            "seed": 42,
            "embedder_max_seq_length": args.max_document_length,
            "reranker_max_length": None,
            "candidate_pool_size": passages_per_query,
            "passage_aggregation": "max_score_per_doc",
            "checkpoint": CHECKPOINT,
            "provenance": provenance,
        },
        retrieval_metrics=retrieval_metrics,
        per_query_results=per_query,
        wall_clock_seconds=sum(latencies_ms) / 1000,
        index_time_seconds=index_timer.elapsed,
        index_size_mb=index_size_mb,
        num_queries=len(qa_items),
        avg_latency_ms=float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
    )
    output_path = RESULTS_DIR / f"{args.output_name}.json"
    result.save(output_path)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
