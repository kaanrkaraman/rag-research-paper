"""Reranker implementations: Azure Cohere, local cross-encoder, FlashRank."""

from __future__ import annotations

import os

import requests

from src.retrieval.base import BaseReranker
from src.utils.common import RetrievedDoc, get_logger

logger = get_logger(__name__)


class AzureCohereReranker(BaseReranker):
    """Cohere Rerank via Azure AI Foundry serverless endpoint."""

    name = "cohere_rerank"

    def __init__(self, model: str = "Cohere-rerank-v4.0-pro", top_n: int = 10):
        self.model = model
        self.top_n = top_n
        self.api_key = os.getenv("AZURE_API_KEY")
        self.endpoint = (
            f"{os.getenv('AZURE_LLM_ENDPOINT', 'https://aif-meftun-academic-work.services.ai.azure.com')}"
            f"/providers/cohere/v2/rerank"
        )

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        if not documents:
            return []

        texts = [d.text[:4096] for d in documents]  # Cohere max 4096 chars per doc
        body = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": min(top_k, len(documents)),
        }
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

        resp = requests.post(self.endpoint, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        reranked = []
        for rank, result in enumerate(data["results"]):
            idx = result["index"]
            orig = documents[idx]
            reranked.append(RetrievedDoc(
                doc_id=orig.doc_id,
                text=orig.text,
                score=result["relevance_score"],
                rank=rank,
                method=f"{orig.method}+cohere_rerank",
                metadata={**orig.metadata, "rerank_score": result["relevance_score"]},
            ))
        return reranked


class LocalCrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker (BGE, MiniLM, etc.)."""

    name = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 max_length: int = 512, batch_size: int = 8):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.name = model_name.split("/")[-1]

        # Auto-cap batch size for non-CUDA devices: MPS hits buffer-size errors
        # on long financial docs (verified) above ~8. CUDA can take 128+.
        try:
            import torch
            on_cuda = torch.cuda.is_available()
        except ImportError:
            on_cuda = False
        if not on_cuda and batch_size > 8:
            logger.info(
                f"LocalCrossEncoderReranker: capping batch_size {batch_size}->8 "
                f"on non-CUDA device (MPS/CPU). Set CUDA to use the configured value."
            )
            batch_size = 8
        self.batch_size = batch_size

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        if not documents:
            return []

        # Truncate doc text to avoid MPS buffer blowup on very long financial docs
        pairs = [[query, d.text[:4096]] for d in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for rank, (doc, score) in enumerate(scored[:top_k]):
            reranked.append(RetrievedDoc(
                doc_id=doc.doc_id,
                text=doc.text,
                score=float(score),
                rank=rank,
                method=f"{doc.method}+{self.name}",
                metadata={**doc.metadata, "rerank_score": float(score)},
            ))
        return reranked


class NoReranker(BaseReranker):
    """Pass-through (no reranking). Used as a baseline."""

    name = "none"

    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        return documents[:top_k]


def create_reranker(config: dict) -> BaseReranker:
    """Factory: create a reranker from config."""
    provider = config.get("provider", "none")
    if provider == "none":
        return NoReranker()
    elif provider in ("cohere", "azure_cohere"):
        return AzureCohereReranker(
            config.get("model", "Cohere-rerank-v4.0-pro"),
            config.get("top_n", 10),
        )
    elif provider == "local":
        return LocalCrossEncoderReranker(
            model_name=config.get("model", "BAAI/bge-reranker-v2-m3"),
            max_length=config.get("max_length", 512),
            batch_size=config.get("batch_size", 8),
        )
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
