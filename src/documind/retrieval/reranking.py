"""Traceable baseline and optional cross-encoder reranking for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, Sequence

from documind.retrieval.hybrid import RetrievalResult
from documind.retrieval.index import LocalIndexManager


@dataclass(frozen=True, slots=True)
class RerankedResult:
    """A reranked candidate retaining its complete original retrieval evidence."""

    retrieval_result: RetrievalResult
    rerank_rank: int
    rerank_score: float
    reranker: str

    @property
    def chunk(self):
        """Expose the source-linked chunk without discarding retrieval provenance."""
        return self.retrieval_result.chunk


class Reranker(Protocol):
    """Rank a bounded set of retrieval candidates for one question."""

    name: str

    def rerank(
        self, question: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RerankedResult]:
        """Return up to ``top_k`` candidates in their new order."""


class IdentityReranker:
    """Transparent baseline: retain retriever order and use its score unchanged."""

    name = "identity"

    def rerank(
        self, question: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RerankedResult]:
        _validate(question, candidates, top_k)
        return [
            RerankedResult(item, rank, float(item.score), self.name)
            for rank, item in enumerate(candidates[:top_k], start=1)
        ]


class CrossEncoderReranker:
    """Optional local cross-encoder; loading is lazy and requires a pinned revision.

    The official Apache-2.0 MS MARCO MiniLM model is the default model name. A
    caller must supply a model revision, or inject a model in tests. No model is
    downloaded by importing this module or constructing the reranker.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

    def __init__(
        self,
        revision: str | None = None,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        if model is None and not revision:
            raise ValueError("Provide a pinned model revision before loading the cross-encoder")
        if not model_name.strip() or not device.strip():
            raise ValueError("model_name and device cannot be empty")
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self._model = model
        self.name = f"cross-encoder:{model_name}@{revision or 'injected'}"

    def rerank(
        self, question: str, candidates: Sequence[RetrievalResult], top_k: int
    ) -> list[RerankedResult]:
        _validate(question, candidates, top_k)
        if not candidates:
            return []
        pairs = [(question, item.chunk.text) for item in candidates]
        scores = self._load_model().predict(pairs, show_progress_bar=False)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = list(scores)
        if len(scores) != len(candidates):
            raise ValueError("Cross-encoder returned a different number of scores than candidates")
        scored = [(item, float(score)) for item, score in zip(candidates, scores, strict=True)]
        scored.sort(key=lambda pair: (-pair[1], pair[0].chunk.chunk_id))
        return [
            RerankedResult(item, rank, score, self.name)
            for rank, (item, score) in enumerate(scored[:top_k], start=1)
        ]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is required for cross-encoder reranking"
                ) from exc
            self._model = CrossEncoder(
                self.model_name,
                revision=self.revision,
                device=self.device,
            )
        return self._model


@dataclass(frozen=True, slots=True)
class RerankRun:
    """Candidate results with separate retrieval and reranking wall times."""

    results: list[RerankedResult]
    retrieval_latency_ms: float
    reranking_latency_ms: float
    candidate_count: int


class RerankingPipeline:
    """Retrieve 10 dense + 10 sparse candidates, fuse, and rerank up to 20."""

    def __init__(self, index: LocalIndexManager, reranker: Reranker) -> None:
        self.index = index
        self.reranker = reranker

    def run(
        self,
        question: str,
        *,
        candidate_k: int = 20,
        top_k: int = 5,
    ) -> RerankRun:
        if not question.strip():
            raise ValueError("question cannot be empty")
        if candidate_k < 1 or candidate_k > 20:
            raise ValueError("candidate_k must be between 1 and 20")
        if top_k < 1 or top_k > candidate_k:
            raise ValueError("top_k must be between 1 and candidate_k")
        started = perf_counter()
        candidates = self.index.search(
            question,
            mode="hybrid",
            top_k=candidate_k,
            dense_k=10,
            sparse_k=10,
        )
        retrieved_at = perf_counter()
        results = self.reranker.rerank(question, candidates, top_k)
        finished = perf_counter()
        return RerankRun(
            results=results,
            retrieval_latency_ms=(retrieved_at - started) * 1000,
            reranking_latency_ms=(finished - retrieved_at) * 1000,
            candidate_count=len(candidates),
        )


def _validate(question: str, candidates: Sequence[RetrievalResult], top_k: int) -> None:
    if not question.strip():
        raise ValueError("question cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
