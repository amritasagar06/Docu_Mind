"""ChromaDB-backed persistent store for canonical DocuMind chunks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from documind.chunking.base import Chunk
from documind.embeddings.base import EmbeddingModel


_SCHEMA_VERSION = 1
_UPSERT_BATCH_SIZE = 2048


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A canonical chunk and its cosine-distance-derived similarity score."""

    chunk: Chunk
    score: float
    distance: float


class ChromaVectorStore:
    """Persist, search, and maintain vectors for source-linked chunks.

    Chroma computes no embeddings itself here: the configured ``EmbeddingModel``
    creates vectors so document and query encoding use the same local model.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        embedder: EmbeddingModel,
        collection_name: str = "documind_chunks",
    ) -> None:
        if len(collection_name) < 3 or len(collection_name) > 63:
            raise ValueError("collection_name must contain 3 to 63 characters")
        if not all(char.isalnum() or char in "_-" for char in collection_name):
            raise ValueError("collection_name may contain only letters, numbers, '_' and '-'")
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.collection_name = collection_name
        self._client = self._make_client()
        self._collection = self._get_or_create(collection_name)
        self._check_model_compatibility(self._collection)

    def add(self, chunks: list[Chunk]) -> int:
        """Add or replace chunks by stable chunk ID; return the count written."""
        self._validate_unique_ids(chunks)
        if not chunks:
            return 0
        self._upsert(self._collection, chunks)
        return len(chunks)

    def query(self, text: str, n_results: int = 5) -> list[SearchResult]:
        """Return nearest source-linked chunks, highest similarity first."""
        return self.query_many([text], n_results=n_results)[0]

    def query_many(self, texts: list[str], n_results: int = 5) -> list[list[SearchResult]]:
        """Search a batch of texts in one embedding and Chroma query call."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("query texts cannot be empty")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        collection_count = self._collection.count()
        if collection_count == 0:
            return [[] for _ in texts]
        vectors = self._embed(texts)
        response = self._collection.query(
            query_embeddings=vectors,
            n_results=min(n_results, collection_count),
            include=["metadatas", "documents", "distances"],
        )
        all_results: list[list[SearchResult]] = []
        for ids, documents, metadatas, distances in zip(
            response.get("ids", []),
            response.get("documents", []),
            response.get("metadatas", []),
            response.get("distances", []),
            strict=True,
        ):
            results: list[SearchResult] = []
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            ):
                dist = float(distance)
                chunk = _chunk_from_metadata(chunk_id, document, metadata)
                results.append(SearchResult(chunk=chunk, score=1.0 - dist, distance=dist))
            all_results.append(results)
        return all_results

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete the supplied chunk IDs and return how many existed."""
        if not chunk_ids:
            return 0
        existing = set(self._collection.get(ids=chunk_ids, include=[]).get("ids", []))
        if existing:
            self._collection.delete(ids=sorted(existing))
        return len(existing)

    def rebuild(self, chunks: list[Chunk]) -> int:
        """Build a replacement collection, then swap it into the active name.

        Existing vectors remain available while embeddings are computed and the
        staging collection is populated. If staging fails, the active collection
        is left intact. The final rename is a short local metadata operation.
        """
        self._validate_unique_ids(chunks)
        stage_name = f"dm_stage_{uuid4().hex}"
        stage = self._get_or_create(stage_name)
        try:
            self._upsert(stage, chunks)
        except Exception:
            self._client.delete_collection(name=stage_name)
            raise

        old_name = f"dm_old_{uuid4().hex}"
        had_active = self.collection_name in self._collection_names()
        try:
            if had_active:
                self._collection.modify(name=old_name)
            stage.modify(name=self.collection_name)
        except Exception:
            if had_active:
                self._client.get_collection(name=old_name, embedding_function=None).modify(
                    name=self.collection_name
                )
            if stage_name in self._collection_names():
                self._client.delete_collection(name=stage_name)
            raise

        if had_active:
            self._client.delete_collection(name=old_name)
        self._collection = self._client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )
        return len(chunks)

    @property
    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self._collection.count()

    def list_chunk_ids(self) -> list[str]:
        """Return all indexed chunk IDs in deterministic order."""
        return sorted(self._collection.get(include=[]).get("ids", []))

    def _upsert(self, collection: Any, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._check_model_compatibility(collection)
        collection_metadata = dict(collection.metadata or {})
        for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
            batch = chunks[start : start + _UPSERT_BATCH_SIZE]
            vectors = self._embed([chunk.text for chunk in batch])
            dimension = len(vectors[0])
            stored_dimension = collection_metadata.get("embedding_dimension")
            if stored_dimension and stored_dimension != dimension and collection.count():
                raise ValueError(
                    f"Collection uses {stored_dimension}-dimensional vectors, but the "
                    f"embedder returned {dimension} dimensions. Rebuild the collection."
                )
            if stored_dimension != dimension:
                collection_metadata["embedding_dimension"] = dimension
                collection_metadata["embedding_model"] = self.embedder.model_name
                collection_metadata["embedding_revision"] = getattr(
                    self.embedder, "revision", "unspecified"
                )
                collection.modify(metadata=_metadata_for_update(collection_metadata))
            collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=vectors,
                documents=[chunk.text for chunk in batch],
                metadatas=[_metadata_for(chunk) for chunk in batch],
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedder.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedder returned a different number of vectors than input texts")
        dimension: int | None = None
        for vector in vectors:
            if not vector or any(not isinstance(value, (int, float)) for value in vector):
                raise ValueError("Embedder must return non-empty numeric vectors")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("Embedder returned inconsistent vector dimensions")
        return vectors

    def _get_or_create(self, name: str) -> Any:
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "documind_schema_version": _SCHEMA_VERSION,
                "embedding_model": self.embedder.model_name,
                "embedding_revision": getattr(self.embedder, "revision", "unspecified"),
            },
        )

    def _check_model_compatibility(self, collection: Any) -> None:
        metadata = collection.metadata or {}
        stored_model = metadata.get("embedding_model")
        configured_revision = getattr(self.embedder, "revision", "unspecified")
        stored_revision = metadata.get("embedding_revision")
        if collection.count() and (
            (stored_model and stored_model != self.embedder.model_name)
            or (stored_revision and stored_revision != configured_revision)
        ):
            raise ValueError(
                "Collection embedding model/revision does not match this store's "
                "configuration. Rebuild into a new collection or use the original model."
            )
        if stored_model != self.embedder.model_name and not collection.count():
            collection.modify(
                metadata=_metadata_for_update(
                    {
                        **metadata,
                        "embedding_model": self.embedder.model_name,
                        "embedding_revision": configured_revision,
                    }
                )
            )

    def _make_client(self) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is missing. Install project dependencies with "
                "`python -m pip install -e \".[dev]\"`."
            ) from exc
        return chromadb.PersistentClient(path=str(self.persist_directory))

    def _collection_names(self) -> set[str]:
        """Support Chroma releases returning either names or Collection objects."""
        return {
            item if isinstance(item, str) else item.name
            for item in self._client.list_collections()
        }

    @staticmethod
    def _validate_unique_ids(chunks: list[Chunk]) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("A batch cannot contain duplicate chunk IDs")


def _metadata_for(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Encode optional and nested canonical metadata into Chroma-safe scalars."""
    metadata: dict[str, str | int | float | bool] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "source_name": chunk.source_name,
        "source_uri": chunk.source_uri,
        "strategy": chunk.strategy,
        "source_headings_json": json.dumps(chunk.source_headings, ensure_ascii=False),
        "custom_metadata_json": json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
    }
    if chunk.heading is not None:
        metadata["heading"] = chunk.heading
    if chunk.page_number is not None:
        metadata["page_number"] = chunk.page_number
    if chunk.content_type is not None:
        metadata["content_type"] = chunk.content_type
    return metadata


def _metadata_for_update(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop Chroma's immutable distance-function setting from metadata edits."""
    return {key: value for key, value in metadata.items() if key != "hnsw:space"}


def _chunk_from_metadata(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    """Reconstruct a complete canonical chunk from persisted Chroma metadata."""
    return Chunk(
        chunk_id=str(chunk_id),
        document_id=str(metadata["document_id"]),
        text=str(text),
        chunk_index=int(metadata["chunk_index"]),
        source_name=str(metadata["source_name"]),
        source_uri=str(metadata["source_uri"]),
        strategy=metadata["strategy"],
        heading=metadata.get("heading"),
        page_number=metadata.get("page_number"),
        content_type=metadata.get("content_type"),
        source_headings=tuple(json.loads(metadata.get("source_headings_json", "[]"))),
        metadata=json.loads(metadata.get("custom_metadata_json", "{}")),
    )
