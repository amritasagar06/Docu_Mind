from documind.chunking import (
    ChunkingConfig,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
)
from documind.ingestion.models import Document


def make_document(text: str, *, page_number: int | None = None) -> Document:
    return Document(
        text=text,
        source_name="guide.md",
        source_uri="file:///docs/guide.md",
        headings=["Setup", "Windows"],
        page_number=page_number,
        content_type="text/markdown",
        metadata={"team": "platform", "revision": 2},
    )


def test_fixed_chunker_respects_size_and_overlap() -> None:
    document = make_document("abcdefghijklmnopqrstuvwxyz")

    chunks = FixedSizeChunker(ChunkingConfig(max_chars=10, overlap_chars=3)).chunk(document)

    assert [len(chunk.text) for chunk in chunks] == [10, 10, 10, 5]
    assert chunks[0].text[-3:] == chunks[1].text[:3]
    assert chunks[1].text[-3:] == chunks[2].text[:3]
    assert all(chunk.strategy == "fixed" for chunk in chunks)


def test_chunks_preserve_source_metadata_and_have_stable_ids() -> None:
    document = make_document("first part second part", page_number=4)
    chunker = FixedSizeChunker(ChunkingConfig(max_chars=12, overlap_chars=2))

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert all(chunk.document_id == first_run[0].document_id for chunk in first_run)
    assert all(chunk.source_name == "guide.md" for chunk in first_run)
    assert all(chunk.source_uri == "file:///docs/guide.md" for chunk in first_run)
    assert all(chunk.page_number == 4 for chunk in first_run)
    assert all(chunk.source_headings == ("Setup", "Windows") for chunk in first_run)
    assert all(chunk.metadata == {"team": "platform", "revision": 2} for chunk in first_run)


def test_recursive_chunker_preserves_markdown_heading_paths() -> None:
    document = make_document(
        "# Setup\nInstall package. Run the command.\n\n"
        "## Windows\nActivate the environment. Verify the import."
    )

    chunks = RecursiveChunker(ChunkingConfig(max_chars=24, overlap_chars=0)).chunk(document)

    assert chunks
    assert all(len(chunk.text) <= 24 for chunk in chunks)
    assert any(chunk.heading == "Setup" for chunk in chunks)
    assert any(chunk.heading == "Setup > Windows" for chunk in chunks)
    assert "Install package" in " ".join(chunk.text for chunk in chunks)
    assert "Activate the environment" in " ".join(chunk.text for chunk in chunks)
    assert all(chunk.strategy == "recursive" for chunk in chunks)


def test_recursive_chunker_splits_unbroken_text_to_the_limit() -> None:
    document = make_document("x" * 29)

    chunks = RecursiveChunker(ChunkingConfig(max_chars=10, overlap_chars=0)).chunk(document)

    assert [len(chunk.text) for chunk in chunks] == [10, 10, 9]
    assert "".join(chunk.text for chunk in chunks) == "x" * 29


def test_semantic_chunker_groups_similar_paragraphs_and_splits_topics() -> None:
    document = make_document(
        "apple setup steps\n\napple setup details\n\norange deployment steps"
    )

    def similarity(left: str, right: str) -> float:
        return 0.9 if "orange" not in right else 0.1

    chunks = SemanticChunker(
        similarity_fn=similarity,
        config=ChunkingConfig(max_chars=100, overlap_chars=0),
        similarity_threshold=0.7,
    ).chunk(document)

    assert len(chunks) == 2
    assert "apple setup steps" in chunks[0].text
    assert "apple setup details" in chunks[0].text
    assert "orange deployment steps" in chunks[1].text
    assert all(chunk.strategy == "semantic" for chunk in chunks)


def test_semantic_chunker_hard_limits_oversized_paragraphs() -> None:
    document = make_document("z" * 31)
    chunker = SemanticChunker(
        similarity_fn=lambda _left, _right: 1.0,
        config=ChunkingConfig(max_chars=12, overlap_chars=0),
    )

    chunks = chunker.chunk(document)

    assert chunks
    assert all(len(chunk.text) <= 12 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == "z" * 31


def test_chunkers_return_no_chunks_for_blank_documents() -> None:
    document = make_document(" \n \t ")

    assert FixedSizeChunker().chunk(document) == []
    assert RecursiveChunker().chunk(document) == []
    assert SemanticChunker(lambda _left, _right: 1.0).chunk(document) == []


def test_chunking_config_rejects_invalid_bounds() -> None:
    for kwargs in (
        {"max_chars": 0},
        {"overlap_chars": -1},
        {"max_chars": 10, "overlap_chars": 10},
    ):
        try:
            ChunkingConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid config to be rejected: {kwargs}")
