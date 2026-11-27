from __future__ import annotations

import pytest

from documind.embeddings import SentenceTransformerEmbedder


class FakeSentenceModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((texts, kwargs))
        return [[0.6, 0.8] for _ in texts]


def test_embedder_uses_lazy_injected_model_and_normalized_vectors() -> None:
    model = FakeSentenceModel()
    embedder = SentenceTransformerEmbedder(model=model)

    vectors = embedder.embed(["first paragraph", "second paragraph"])

    assert vectors == [[0.6, 0.8], [0.6, 0.8]]
    texts, options = model.calls[0]
    assert texts == ["first paragraph", "second paragraph"]
    assert options["normalize_embeddings"] is True
    assert options["show_progress_bar"] is False
    assert "device" not in options
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedder_returns_empty_for_empty_batch_without_loading_model() -> None:
    embedder = SentenceTransformerEmbedder()

    assert embedder.embed([]) == []
    assert embedder._model is None


@pytest.mark.parametrize("texts", [[""], ["  "], ["valid", "\n"]])
def test_embedder_rejects_blank_text(texts: list[str]) -> None:
    embedder = SentenceTransformerEmbedder(model=FakeSentenceModel())

    with pytest.raises(ValueError, match="non-empty string"):
        embedder.embed(texts)


def test_embedder_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=0)
