from __future__ import annotations

import pytest

from documind.chunking import FixedSizeChunker
from documind.ingestion.models import Document
from documind.retrieval import (
    CrossEncoderReranker,
    IdentityReranker,
    RetrievalResult,
    RerankingPipeline,
)


def make_results() -> list[RetrievalResult]:
    chunker = FixedSizeChunker()
    chunks = [
        chunker.chunk(
            Document(
                text=text,
                source_name="guide.md",
                source_uri=f"file:///guide-{index}.md",
            )
        )[0]
        for index, text in enumerate(
            ("First passage about a database pool.", "Second passage about a pool limit.")
        )
    ]
    return [
        RetrievalResult(chunk, "hybrid", score, dense_rank=i + 1, rrf_score=score)
        for i, (chunk, score) in enumerate(zip(chunks, (0.03, 0.02), strict=True))
    ]


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        return [2.0, 5.0]


def test_identity_reranker_preserves_order_scores_and_sources() -> None:
    candidates = make_results()
    reranked = IdentityReranker().rerank("database pool", candidates, top_k=1)
    assert len(reranked) == 1
    assert reranked[0].chunk.chunk_id == candidates[0].chunk.chunk_id
    assert reranked[0].rerank_score == candidates[0].score
    assert reranked[0].retrieval_result.dense_rank == 1
    assert reranked[0].reranker == "identity"


def test_cross_encoder_reranks_and_keeps_retrieval_provenance() -> None:
    candidates = make_results()
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())
    result = reranker.rerank("Which passage?", candidates, top_k=2)
    assert [item.chunk.chunk_id for item in result] == [
        candidates[1].chunk.chunk_id,
        candidates[0].chunk.chunk_id,
    ]
    assert [item.rerank_rank for item in result] == [1, 2]
    assert result[0].rerank_score == 5.0
    assert result[0].retrieval_result.rrf_score == candidates[1].rrf_score
    assert result[0].chunk.source_name == "guide.md"


def test_cross_encoder_requires_a_pinned_revision_for_real_loading() -> None:
    with pytest.raises(ValueError, match="pinned model revision"):
        CrossEncoderReranker()


def test_cross_encoder_checks_score_count() -> None:
    class BadModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.5]

    with pytest.raises(ValueError, match="different number of scores"):
        CrossEncoderReranker(model=BadModel()).rerank("question", make_results(), 2)


def test_pipeline_uses_hybrid_candidates_and_returns_separate_timings() -> None:
    class FakeIndex:
        def __init__(self):
            self.options = None

        def search(self, question, **kwargs):
            self.options = (question, kwargs)
            return make_results()

    index = FakeIndex()
    run = RerankingPipeline(index, IdentityReranker()).run("a question")
    assert index.options[1] == {
        "mode": "hybrid",
        "top_k": 20,
        "dense_k": 10,
        "sparse_k": 10,
    }
    assert run.candidate_count == 2
    assert run.results[0].chunk.chunk_id == make_results()[0].chunk.chunk_id
    assert run.retrieval_latency_ms >= 0
    assert run.reranking_latency_ms >= 0


@pytest.mark.parametrize("candidate_k,top_k", [(0, 1), (21, 1), (2, 3)])
def test_pipeline_rejects_invalid_candidate_or_output_limits(candidate_k, top_k) -> None:
    with pytest.raises(ValueError):
        RerankingPipeline(object(), IdentityReranker()).run(
            "question", candidate_k=candidate_k, top_k=top_k
        )
