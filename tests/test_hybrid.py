from __future__ import annotations

from pathlib import Path

import pytest

from documind.chunking import FixedSizeChunker
from documind.ingestion.models import Document
from documind.retrieval import (
    BM25Result,
    LocalIndexManager,
    ReciprocalRankFusion,
    RetrievalResult,
)
from documind.vectorstore import SearchResult as DenseResult


pytest.importorskip("chromadb", reason="Install project dependencies for retrieval tests")
pytest.importorskip("rank_bm25", reason="Install project dependencies for BM25 tests")


class TopicEmbedder:
    model_name = "tests/hybrid-embedder"
    revision = "v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0]
            if "pool" in text.casefold() or "max_pool_size" in text.casefold()
            else [0.0, 1.0, 0.0]
            for text in texts
        ]


def make_chunk(text: str, name: str):
    return FixedSizeChunker().chunk(
        Document(
            text=text,
            source_name=name,
            source_uri=f"file:///docs/{name}",
            headings=["Configuration"],
            content_type="text/markdown",
            metadata={"team": "platform"},
        )
    )[0]


def test_rrf_fuses_overlap_and_preserves_channel_ranks_and_scores() -> None:
    first = make_chunk("Set the MAX_POOL_SIZE option.", "first.md")
    overlap = make_chunk("The pool limit is configured here.", "overlap.md")
    sparse_only = make_chunk("MAX_POOL_SIZE must be configured.", "sparse.md")
    dense = [
        DenseResult(first, score=0.9, distance=0.1),
        DenseResult(overlap, score=0.8, distance=0.2),
    ]
    sparse = [BM25Result(overlap, score=3.2), BM25Result(sparse_only, score=1.4)]

    results = ReciprocalRankFusion(constant=60).fuse(dense, sparse, top_k=3)

    assert [item.chunk.chunk_id for item in results] == [
        overlap.chunk_id,
        first.chunk_id,
        sparse_only.chunk_id,
    ]
    best = results[0]
    assert best.mode == "hybrid"
    assert best.dense_rank == 2
    assert best.sparse_rank == 1
    assert best.dense_score == pytest.approx(0.8)
    assert best.sparse_score == pytest.approx(3.2)
    assert best.rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert best.score == best.rrf_score


def test_rrf_handles_missing_lists_duplicate_ids_and_ties_deterministically() -> None:
    first = make_chunk("One technical note.", "first.md")
    second = make_chunk("Another technical note.", "second.md")
    ranker = ReciprocalRankFusion(constant=10)

    dense_only = ranker.fuse(
        [
            DenseResult(second, score=0.7, distance=0.3),
            DenseResult(second, score=0.6, distance=0.4),
            DenseResult(first, score=0.5, distance=0.5),
        ],
        [],
        top_k=5,
    )
    assert {item.chunk.chunk_id for item in dense_only} == {first.chunk_id, second.chunk_id}
    second_result = next(item for item in dense_only if item.chunk.chunk_id == second.chunk_id)
    assert second_result.dense_rank == 1
    assert second_result.sparse_rank is None
    assert second_result.rrf_score == pytest.approx(1 / 11)
    assert ranker.fuse([], [], top_k=5) == []

    tied = ranker.fuse(
        [
            DenseResult(second, score=0.7, distance=0.3),
            DenseResult(first, score=0.6, distance=0.4),
        ],
        [BM25Result(first, 2.0), BM25Result(second, 1.0)],
        top_k=5,
    )
    assert [item.chunk.chunk_id for item in tied] == sorted(
        [first.chunk_id, second.chunk_id]
    )

    sparse_only = ranker.fuse([], [BM25Result(first, 1.0)], top_k=1)
    assert len(sparse_only) == 1
    assert sparse_only[0].dense_rank is None
    assert sparse_only[0].sparse_rank == 1


def test_local_search_compares_dense_sparse_and_hybrid_modes(tmp_path: Path) -> None:
    index = LocalIndexManager(tmp_path / "index", TopicEmbedder())
    pool = make_chunk("Set MAX_POOL_SIZE to control the connection pool.", "pool.md")
    other = make_chunk("Restart the service after updating configuration.", "restart.md")
    index.rebuild([pool, other])

    question = "How do I change MAX_POOL_SIZE?"
    dense = index.search(question, mode="dense", top_k=2)
    sparse = index.search(question, mode="sparse", top_k=2)
    hybrid = index.search(
        question,
        mode="hybrid",
        top_k=2,
        dense_k=2,
        sparse_k=2,
    )

    assert all(result.chunk.chunk_id in {pool.chunk_id, other.chunk_id} for result in dense)
    assert all(result.chunk.chunk_id in {pool.chunk_id, other.chunk_id} for result in sparse)
    assert all(result.chunk.chunk_id in {pool.chunk_id, other.chunk_id} for result in hybrid)
    assert dense[0].chunk.chunk_id == pool.chunk_id
    assert sparse[0].chunk.chunk_id == pool.chunk_id
    assert hybrid[0].chunk.chunk_id == pool.chunk_id
    assert dense[0].dense_rank == 1 and dense[0].rrf_score is None
    assert sparse[0].sparse_rank == 1 and sparse[0].rrf_score is None
    assert hybrid[0].dense_rank == 1 and hybrid[0].sparse_rank == 1
    assert hybrid[0].rrf_score == pytest.approx(1 / 61 + 1 / 61)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"constant": 0}, "constant"),
    ],
)
def test_rrf_rejects_invalid_configuration(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ReciprocalRankFusion(**kwargs)


def test_hybrid_retriever_validates_modes_and_top_k(tmp_path: Path) -> None:
    index = LocalIndexManager(tmp_path / "index", TopicEmbedder())
    with pytest.raises(ValueError, match="question"):
        index.search("  ")
    with pytest.raises(ValueError, match="mode"):
        index.search("pool", mode="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k"):
        index.search("pool", top_k=0)
    with pytest.raises(ValueError, match="dense_k"):
        index.search("pool", dense_k=0)


def test_rrf_constant_is_configurable_through_local_manager(tmp_path: Path) -> None:
    index = LocalIndexManager(tmp_path / "index", TopicEmbedder(), rrf_constant=17)

    assert index.retriever.fusion.constant == 17
