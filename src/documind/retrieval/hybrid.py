"""Dense, sparse, and Reciprocal Rank Fusion retrieval interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from documind.chunking.base import Chunk
from documind.retrieval.bm25 import BM25Result, BM25SearchIndex
from documind.vectorstore.chroma import ChromaVectorStore, SearchResult as DenseResult


RetrievalMode = Literal["dense", "sparse", "hybrid"]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Source-linked result with per-retriever rank and score provenance."""

    chunk: Chunk
    mode: RetrievalMode
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None


class ReciprocalRankFusion:
    """Combine two ranked result lists with deterministic RRF ordering."""

    def __init__(self, constant: int = 60) -> None:
        if constant < 1:
            raise ValueError("constant must be at least 1")
        self.constant = constant

    def fuse(
        self,
        dense_results: list[DenseResult],
        sparse_results: list[BM25Result],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Fuse source-linked results; missing channels contribute no score.

        A chunk's rank is its one-based position in the supplied list. Repeated
        IDs in one list use only their first occurrence, preventing duplicate
        rows from gaining extra RRF weight.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        by_id: dict[str, dict[str, object]] = {}
        for rank, result in enumerate(dense_results, start=1):
            entry = by_id.setdefault(result.chunk.chunk_id, {"chunk": result.chunk})
            if "dense_rank" not in entry:
                entry["dense_rank"] = rank
                entry["dense_score"] = float(result.score)
        for rank, result in enumerate(sparse_results, start=1):
            entry = by_id.setdefault(result.chunk.chunk_id, {"chunk": result.chunk})
            if "sparse_rank" not in entry:
                entry["sparse_rank"] = rank
                entry["sparse_score"] = float(result.score)

        fused: list[RetrievalResult] = []
        for entry in by_id.values():
            dense_rank = entry.get("dense_rank")
            sparse_rank = entry.get("sparse_rank")
            rrf_score = 0.0
            if dense_rank is not None:
                rrf_score += 1.0 / (self.constant + int(dense_rank))
            if sparse_rank is not None:
                rrf_score += 1.0 / (self.constant + int(sparse_rank))
            fused.append(
                RetrievalResult(
                    chunk=entry["chunk"],  # type: ignore[arg-type]
                    mode="hybrid",
                    score=rrf_score,
                    dense_rank=int(dense_rank) if dense_rank is not None else None,
                    sparse_rank=int(sparse_rank) if sparse_rank is not None else None,
                    dense_score=(
                        float(entry["dense_score"]) if "dense_score" in entry else None
                    ),
                    sparse_score=(
                        float(entry["sparse_score"]) if "sparse_score" in entry else None
                    ),
                    rrf_score=rrf_score,
                )
            )
        fused.sort(
            key=lambda result: (
                -result.score,
                min(
                    rank
                    for rank in (result.dense_rank, result.sparse_rank)
                    if rank is not None
                ),
                result.chunk.chunk_id,
            )
        )
        return fused[:top_k]


class HybridRetriever:
    """Run dense-only, sparse-only, or fused retrieval over shared chunks."""

    def __init__(
        self,
        dense_index: ChromaVectorStore,
        sparse_index: BM25SearchIndex,
        rrf_constant: int = 60,
    ) -> None:
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.fusion = ReciprocalRankFusion(rrf_constant)

    def retrieve(
        self,
        question: str,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve the same question through a selected ranking strategy."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError("mode must be 'dense', 'sparse', or 'hybrid'")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if dense_k < 1 or sparse_k < 1:
            raise ValueError("dense_k and sparse_k must be at least 1")

        dense_results = (
            self.dense_index.query(question, n_results=dense_k)
            if mode in {"dense", "hybrid"}
            else []
        )
        sparse_results = (
            self.sparse_index.query(question, n_results=sparse_k)
            if mode in {"sparse", "hybrid"}
            else []
        )

        if mode == "hybrid":
            return self.fusion.fuse(dense_results, sparse_results, top_k=top_k)
        if mode == "dense":
            return [
                RetrievalResult(
                    chunk=result.chunk,
                    mode="dense",
                    score=result.score,
                    dense_rank=rank,
                    dense_score=result.score,
                )
                for rank, result in enumerate(dense_results[:top_k], start=1)
            ]
        return [
            RetrievalResult(
                chunk=result.chunk,
                mode="sparse",
                score=result.score,
                sparse_rank=rank,
                sparse_score=result.score,
            )
            for rank, result in enumerate(sparse_results[:top_k], start=1)
        ]
