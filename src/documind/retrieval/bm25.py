"""BM25 keyword retrieval with canonical chunks as its rebuild source."""

from __future__ import annotations

from dataclasses import dataclass
import re

from documind.chunking.base import Chunk
from documind.retrieval.manifest import ChunkManifestStore


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[_./:+-][a-z0-9]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Split text while retaining common technical identifiers as single terms."""
    return [match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True, slots=True)
class BM25Result:
    """A canonical source chunk and its BM25 relevance score."""

    chunk: Chunk
    score: float


class BM25SearchIndex:
    """Rebuild an in-memory BM25 ranking index from a persisted chunk manifest."""

    def __init__(self, manifest: ChunkManifestStore) -> None:
        self.manifest = manifest
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the ranker from the canonical JSONL manifest."""
        self._chunks = self.manifest.load()
        self._tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        self._bm25 = None
        if self._chunks and any(self._tokenized):
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise RuntimeError(
                    "rank-bm25 is missing. Install project dependencies with "
                    '`python -m pip install -e ".[dev]"`.'
                ) from exc
            self._bm25 = BM25Okapi(self._tokenized)

    def query(self, text: str, n_results: int = 5) -> list[BM25Result]:
        """Return source-linked chunks containing query terms, ranked by BM25."""
        if not text.strip():
            raise ValueError("query text cannot be empty")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        terms = tokenize(text)
        if not terms or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(terms)
        candidates = [
            (index, float(score))
            for index, (chunk, document_terms, score) in enumerate(
                zip(self._chunks, self._tokenized, scores, strict=True)
            )
            if set(terms).intersection(document_terms)
        ]
        candidates.sort(key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        return [
            BM25Result(chunk=self._chunks[index], score=score)
            for index, score in candidates[:n_results]
        ]

    @property
    def count(self) -> int:
        """Return the number of canonical chunks in the BM25 corpus."""
        return len(self._chunks)
