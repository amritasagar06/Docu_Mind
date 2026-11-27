from __future__ import annotations

from pathlib import Path

import pytest

from documind.chunking import FixedSizeChunker
from documind.ingestion.models import Document
from documind.retrieval import (
    ChunkManifestStore,
    DuplicateStatus,
    LocalIndexManager,
    tokenize,
)


pytest.importorskip("chromadb", reason="Install project dependencies for index tests")
pytest.importorskip("rank_bm25", reason="Install project dependencies for BM25 tests")


class TopicEmbedder:
    model_name = "tests/topic-embedder"
    revision = "v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "max_pool_size" in lowered or "pool capacity" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "mqrc_cluster_resolution_error" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def make_chunk(text: str, name: str = "guide.md"):
    return FixedSizeChunker().chunk(
        Document(
            text=text,
            source_name=name,
            source_uri=f"file:///docs/{name}",
            headings=["Troubleshooting"],
            content_type="text/markdown",
            metadata={"team": "platform"},
        )
    )[0]


def test_tokenizer_preserves_technical_identifiers() -> None:
    assert tokenize("MAX_POOL_SIZE caused MQRC_CLUSTER_RESOLUTION_ERROR (HTTP 422)") == [
        "max_pool_size",
        "caused",
        "mqrc_cluster_resolution_error",
        "http",
        "422",
    ]


def test_bm25_finds_exact_identifier_and_error_code(tmp_path: Path) -> None:
    manifest = ChunkManifestStore(tmp_path / "chunks.jsonl")
    chunks = [
        make_chunk("Set MAX_POOL_SIZE to control the maximum pool capacity.", "pool.md"),
        make_chunk(
            "MQRC_CLUSTER_RESOLUTION_ERROR can appear when resolving a cluster.",
            "mq.md",
        ),
        make_chunk("Restart the service after changing its configuration.", "restart.md"),
    ]
    manifest.replace(chunks)
    from documind.retrieval import BM25SearchIndex

    index = BM25SearchIndex(manifest)

    pool_result = index.query("MAX_POOL_SIZE")
    error_result = index.query("MQRC_CLUSTER_RESOLUTION_ERROR")
    assert pool_result[0].chunk.chunk_id == chunks[0].chunk_id
    assert error_result[0].chunk.chunk_id == chunks[1].chunk_id
    assert index.query("identifier_not_present") == []


def test_local_manager_persists_and_syncs_both_indexes(tmp_path: Path) -> None:
    manager = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())
    pool = make_chunk("Set MAX_POOL_SIZE to control pool capacity.", "pool.md")
    error = make_chunk(
        "MQRC_CLUSTER_RESOLUTION_ERROR indicates a cluster lookup problem.",
        "mq.md",
    )

    assert manager.add([pool, error]) == 2
    assert manager.vector_store.list_chunk_ids() == sorted([pool.chunk_id, error.chunk_id])
    assert manager.keyword_index.count == manager.vector_store.count == 2
    assert manager.keyword_index.query("MAX_POOL_SIZE")[0].chunk.chunk_id == pool.chunk_id

    reopened = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())
    assert [chunk.chunk_id for chunk in reopened.chunks()] == sorted(
        [pool.chunk_id, error.chunk_id]
    )
    assert reopened.keyword_index.query("MQRC_CLUSTER_RESOLUTION_ERROR")[0].chunk.chunk_id == error.chunk_id
    assert reopened.delete([pool.chunk_id]) == 1
    assert reopened.vector_store.list_chunk_ids() == [error.chunk_id]
    assert reopened.keyword_index.count == 1
    assert reopened.keyword_index.query("MAX_POOL_SIZE") == []

    replacement = make_chunk("Set MAX_POOL_SIZE to control the new pool capacity.", "new.md")
    assert reopened.rebuild([replacement]) == 1
    assert reopened.vector_store.list_chunk_ids() == [replacement.chunk_id]
    assert reopened.keyword_index.count == 1
    assert reopened.keyword_index.query("MAX_POOL_SIZE")[0].chunk.chunk_id == replacement.chunk_id


def test_manager_repairs_stale_vector_ids_from_manifest(tmp_path: Path) -> None:
    manager = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())
    chunk = make_chunk("Set MAX_POOL_SIZE to control pool capacity.")
    manager.add([chunk])
    manager.vector_store.delete([chunk.chunk_id])

    repaired = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())

    assert repaired.vector_store.list_chunk_ids() == [chunk.chunk_id]
    assert repaired.keyword_index.query("MAX_POOL_SIZE")[0].chunk.chunk_id == chunk.chunk_id


def test_duplicate_flags_are_batched_reviewable_and_reversible(tmp_path: Path) -> None:
    manager = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())
    first = make_chunk("Set MAX_POOL_SIZE to control the connection pool.", "first.md")
    near_copy = make_chunk("Set MAX_POOL_SIZE to control the connection pool.", "copy.md")
    distinct = make_chunk("Restart the service after changing its configuration.", "other.md")
    manager.rebuild([first, near_copy, distinct])

    candidates = manager.detect_duplicates(threshold=0.95)
    assert [(item.first_chunk_id, item.second_chunk_id) for item in candidates] == [
        tuple(sorted([first.chunk_id, near_copy.chunk_id]))
    ]
    assert candidates[0].similarity == pytest.approx(1.0)

    registry = manager.duplicate_registry
    confirmed = manager.set_duplicate_status(
        first.chunk_id,
        near_copy.chunk_id,
        DuplicateStatus.DUPLICATE,
    )
    assert confirmed.status == DuplicateStatus.DUPLICATE
    assert len(manager.chunks()) == 3

    registry.record_candidates(candidates)
    assert registry.all()[0].status == DuplicateStatus.DUPLICATE
    reversed_decision = manager.set_duplicate_status(
        first.chunk_id,
        near_copy.chunk_id,
        DuplicateStatus.DISTINCT,
    )
    assert reversed_decision.status == DuplicateStatus.DISTINCT
    assert len(manager.chunks()) == 3

    reopened = LocalIndexManager(tmp_path / "local-index", TopicEmbedder())
    assert reopened.duplicate_registry.all()[0].status == DuplicateStatus.DISTINCT
    assert len(reopened.chunks()) == 3
