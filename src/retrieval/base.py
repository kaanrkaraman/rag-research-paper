"""Abstract base classes for retrievers and embedders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.utils.common import RetrievedDoc, get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract embedder interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a list of documents. Returns (N, D) array."""
        ...

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed a list of queries. Returns (N, D) array."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class BaseRetriever(ABC):
    """Abstract retriever interface."""

    name: str = "base"

    @abstractmethod
    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build the retrieval index from a corpus."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve top-k documents for a query."""
        ...

    def retrieve_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievedDoc]]:
        """Retrieve for a batch of queries. Default: sequential."""
        return [self.retrieve(q, top_k) for q in queries]


class BaseReranker(ABC):
    """Abstract reranker interface."""

    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, documents: list[RetrievedDoc], top_k: int = 5
    ) -> list[RetrievedDoc]:
        """Rerank documents for a query."""
        ...


class AzureOpenAIEmbedder(BaseEmbedder):
    """Azure OpenAI API-based embedder."""

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 3072):
        import os
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_EMBED_ENDPOINT"),
        )
        self.model = model
        self._dimension = dimensions
        logger.info(f"AzureOpenAIEmbedder: {model}, dim={dimensions}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(
                input=batch, model=self.model, dimensions=self._dimension
            )
            all_embeddings.extend([e.embedding for e in resp.data])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries)


class CohereEmbedder(BaseEmbedder):
    """Cohere API-based embedder with query/document input types."""

    def __init__(self, model: str = "embed-v4.0", dimensions: int = 1024):
        import cohere
        self.client = cohere.ClientV2()
        self.model = model
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(
        self, texts: list[str], input_type: str, batch_size: int = 32
    ) -> np.ndarray:
        import time as _time
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(5):
                try:
                    resp = self.client.embed(
                        texts=batch,
                        model=self.model,
                        input_type=input_type,
                        embedding_types=["float"],
                    )
                    all_embeddings.extend(resp.embeddings.float_)
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = 60 * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                        _time.sleep(wait)
                    else:
                        raise
            # Pace requests to stay under 100k tokens/min
            if i + batch_size < len(texts):
                _time.sleep(3.0)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, input_type="search_document")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries, input_type="search_query")


class VoyageEmbedder(BaseEmbedder):
    """Voyage AI API-based embedder."""

    def __init__(self, model: str = "voyage-3-large", dimensions: int = 1024):
        import voyageai
        self.client = voyageai.Client()
        self.model = model
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(
        self, texts: list[str], input_type: str, batch_size: int = 128
    ) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embed(batch, model=self.model, input_type=input_type)
            all_embeddings.extend(resp.embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, input_type="document")

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries, input_type="query")


class LocalEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder (BGE-M3, E5, etc.).

    Default ``max_seq_length=8192`` matches BGE-M3's native maximum and the
    ~8191-token context that ``text-embedding-3-large`` accepts via the
    OpenAI API. Pass ``max_seq_length`` via config to override (e.g. 512 for
    bge-large-en or 4096 for tighter memory).

    Batch sizes auto-scale by device and sequence length:
      - CUDA, max_seq_length > 4096: docs=8, queries=128
      - CUDA, max_seq_length <= 4096: docs=64, queries=128
      - MPS / CPU: docs=8, queries=32
    Sentence-transformers pads each batch to the longest sample in that
    batch (not to ``max_seq_length``), so most batches over the
    T²-RAGBench corpus run well below peak memory.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", max_seq_length: int = 8192):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # Don't exceed the model's native maximum (e.g. bge-large-en caps at 512).
        # Most modern dense encoders have native maxes set on load:
        #   BGE-M3: 8192, text-embedding-3-large: 8191 (API), bge-large-en: 512.
        native_max = getattr(self.model, "max_seq_length", max_seq_length)
        self.model.max_seq_length = min(max_seq_length, native_max)
        self._dimension = self.model.get_sentence_embedding_dimension()

        # Auto-scale batch sizes for the active device. Long sequences (>4096
        # tokens) get a smaller docs_batch on CUDA so attention activations stay
        # within the memory budget on a 40-100 GB datacenter card.
        try:
            import torch
            self._on_cuda = torch.cuda.is_available()
        except ImportError:
            self._on_cuda = False
        if self._on_cuda:
            self._docs_batch = 8 if self.model.max_seq_length > 4096 else 64
        else:
            self._docs_batch = 8
        self._queries_batch = 128 if self._on_cuda else 32

        logger.info(
            f"LocalEmbedder: {model_name}, dim={self._dimension}, "
            f"max_seq={self.model.max_seq_length}, "
            f"docs_batch={self._docs_batch}, queries_batch={self._queries_batch}"
        )

    @property
    def max_seq_length(self) -> int:
        """The actual max_seq_length applied at encode time (post-cap)."""
        return int(self.model.max_seq_length)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=self._docs_batch,
        )

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode(
            queries,
            show_progress_bar=len(queries) > 16,
            normalize_embeddings=True,
            batch_size=self._queries_batch,
        )


def create_embedder(config: dict) -> BaseEmbedder:
    """Factory: create an embedder from config."""
    provider = config.get("provider", "azure")
    if provider in ("openai", "azure"):
        return AzureOpenAIEmbedder(config["model"], config.get("dimensions", 3072))
    elif provider == "cohere":
        return CohereEmbedder(config["model"], config.get("dimensions", 1024))
    elif provider == "voyage":
        return VoyageEmbedder(config["model"], config.get("dimensions", 1024))
    elif provider == "local":
        return LocalEmbedder(
            config["model"],
            max_seq_length=config.get("max_seq_length", 8192),
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
