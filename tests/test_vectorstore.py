from __future__ import annotations

from pathlib import Path

import pytest

from documind.chunking import FixedSizeChunker
from documind.ingestion.models import Document


chromadb = pytest.importorskip("chromadb", reason="Install project dependencies for Chroma tests")


class TinyEmbedder:
    """Deterministic test vectors; avoids downloading a language model."""

    model_name = "tests/tiny-2d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        output = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in ("database", "connection", "pool")):
                output.append([1.0, 0.0])
            else:
                output.append([0.0, 1.0])
        return output


def make_chunks():
    doc = Document(
        text="# Database\n\nConfigure the database connection pool.\n\n"
        "# Other\n\nUnrelated gardening notes.",
        source_name="guide.md",
        source_uri="file:///docs/guide.md",
        headings=["Database", "Other"],
        content_type="text/markdown",
        metadata={"team": "platform", "reviewed": True},
    )
    return FixedSizeChunker().chunk(doc)


def test_add_query_reopen_and_delete_preserve_chunk_provenance(tmp_path: Path) -> None:
    from documind.vectorstore import ChromaVectorStore

    chunks = make_chunks()
    store = ChromaVectorStore(tmp_path / "index", TinyEmbedder())
    assert store.add(chunks) == len(chunks)
    assert store.count == len(chunks)

    results = store.query("database connection pool", n_results=1)
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].chunk.document_id == chunks[0].document_id
    assert results[0].chunk.text == chunks[0].text
    assert results[0].chunk.source_uri == chunks[0].source_uri
    assert results[0].chunk.content_type == "text/markdown"
    assert results[0].chunk.source_headings == ("Database", "Other")
    assert results[0].chunk.metadata == {"team": "platform", "reviewed": True}

    reopened = ChromaVectorStore(tmp_path / "index", TinyEmbedder())
    assert reopened.count == len(chunks)
    assert reopened.query("database pool", n_results=1)[0].chunk.chunk_id == chunks[0].chunk_id
    assert reopened.delete([chunks[0].chunk_id, "missing-id"]) == 1
    assert reopened.count == len(chunks) - 1


def test_rebuild_swaps_contents_and_empty_rebuild_clears_index(tmp_path: Path) -> None:
    from documind.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(tmp_path / "index", TinyEmbedder())
    original = make_chunks()
    assert store.add(original) == len(original)

    replacement = FixedSizeChunker().chunk(
        Document(
            text="A new database guide.",
            source_name="new.md",
            source_uri="file:///docs/new.md",
        )
    )
    assert store.rebuild(replacement) == len(replacement)
    assert store.count == len(replacement)
    assert {item.chunk.chunk_id for item in store.query("database", 5)} == {
        chunk.chunk_id for chunk in replacement
    }

    assert store.rebuild([]) == 0
    assert store.count == 0
    assert store.query("database") == []


def test_query_and_duplicate_ids_are_validated(tmp_path: Path) -> None:
    from documind.vectorstore import ChromaVectorStore

    store = ChromaVectorStore(tmp_path / "index", TinyEmbedder())
    with pytest.raises(ValueError, match="query text"):
        store.query("  ")
    with pytest.raises(ValueError, match="n_results"):
        store.query("database", 0)

    chunk = make_chunks()[0]
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        store.add([chunk, chunk])
