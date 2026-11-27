"""Coordinate a canonical chunk manifest with dense and sparse indexes."""

from __future__ import annotations

from pathlib import Path

from documind.chunking.base import Chunk
from documind.embeddings.base import EmbeddingModel
from documind.retrieval.bm25 import BM25SearchIndex
from documind.retrieval.duplicates import (
    DuplicateCandidate,
    DuplicateDetector,
    DuplicateRegistry,
    DuplicateStatus,
)
from documind.retrieval.hybrid import HybridRetriever, RetrievalMode, RetrievalResult
from documind.retrieval.manifest import ChunkManifestStore
from documind.vectorstore import ChromaVectorStore


class LocalIndexManager:
    """Keep canonical chunks, BM25, and Chroma derived from one manifest.

    The JSONL manifest is the source of truth. A manager opened against an
    existing manifest checks the Chroma IDs and repairs a stale vector index by
    rebuilding it from that manifest. It refuses to erase a populated Chroma
    collection when no manifest exists to explain its contents.
    """

    def __init__(
        self,
        directory: str | Path,
        embedder: EmbeddingModel,
        collection_name: str = "documind_chunks",
        rrf_constant: int = 60,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = ChunkManifestStore(self.directory / "chunk_manifest.jsonl")
        self.vector_store = ChromaVectorStore(
            self.directory / "chroma",
            embedder,
            collection_name=collection_name,
        )
        chunks = self.manifest.load()
        if not self.manifest.path.exists() and self.vector_store.count:
            raise ValueError(
                "A populated Chroma collection exists without a canonical chunk manifest. "
                "Choose a new index directory or create/review the manifest before syncing."
            )
        if self.vector_store.list_chunk_ids() != sorted(chunk.chunk_id for chunk in chunks):
            self.vector_store.rebuild(chunks)
        self.keyword_index = BM25SearchIndex(self.manifest)
        self.duplicate_registry = DuplicateRegistry(self.directory / "duplicate_reviews.jsonl")
        self.retriever = HybridRetriever(
            self.vector_store,
            self.keyword_index,
            rrf_constant=rrf_constant,
        )

    def add(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the manifest and both derived indexes."""
        _validate_unique_ids(chunks)
        if not chunks:
            return 0
        previous = self.manifest.load()
        try:
            self.vector_store.add(chunks)
            self.manifest.upsert(chunks)
        except Exception:
            self.vector_store.rebuild(previous)
            raise
        self.keyword_index.refresh()
        return len(chunks)

    def delete(self, chunk_ids: list[str]) -> int:
        """Remove IDs from the canonical manifest, BM25 corpus, and Chroma."""
        previous = self.manifest.load()
        requested = set(chunk_ids)
        removed = [chunk for chunk in previous if chunk.chunk_id in requested]
        if not removed:
            return 0
        remaining = [chunk for chunk in previous if chunk.chunk_id not in requested]
        self.vector_store.delete([chunk.chunk_id for chunk in removed])
        try:
            self.manifest.replace(remaining)
        except Exception:
            self.vector_store.rebuild(previous)
            raise
        self.keyword_index.refresh()
        return len(removed)

    def rebuild(self, chunks: list[Chunk]) -> int:
        """Replace canonical chunks and both indexes from one reviewed set."""
        _validate_unique_ids(chunks)
        previous = self.manifest.load()
        self.vector_store.rebuild(chunks)
        try:
            self.manifest.replace(chunks)
        except Exception:
            self.vector_store.rebuild(previous)
            raise
        self.keyword_index.refresh()
        return len(chunks)

    def chunks(self) -> list[Chunk]:
        """Return the canonical manifest contents in stable ID order."""
        return self.manifest.load()

    def detect_duplicates(
        self,
        threshold: float = 0.95,
        candidates_per_chunk: int = 10,
    ) -> list[DuplicateCandidate]:
        """Detect and persist likely duplicate pairs without filtering chunks."""
        detector = DuplicateDetector(threshold, candidates_per_chunk)
        candidates = detector.detect(self.manifest.load(), self.vector_store)
        self.duplicate_registry.record_candidates(candidates)
        return candidates

    def set_duplicate_status(
        self,
        first_chunk_id: str,
        second_chunk_id: str,
        status: DuplicateStatus,
    ) -> DuplicateCandidate:
        """Record or reverse a review decision for a detected pair."""
        return self.duplicate_registry.set_status(first_chunk_id, second_chunk_id, status)

    def search(
        self,
        question: str,
        *,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10,
    ) -> list[RetrievalResult]:
        """Search using dense, sparse, or hybrid ranking over the same manifest."""
        return self.retriever.retrieve(
            question,
            mode=mode,
            top_k=top_k,
            dense_k=dense_k,
            sparse_k=sparse_k,
        )


def _validate_unique_ids(chunks: list[Chunk]) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("A batch cannot contain duplicate chunk IDs")
