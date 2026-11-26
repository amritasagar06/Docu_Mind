"""Shared chunk model, configuration, and chunk identity helpers."""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal, Protocol

from documind.ingestion.models import Document


ChunkStrategy = Literal["fixed", "recursive", "semantic"]


@dataclass(slots=True)
class Chunk:
    """A searchable text segment with traceable document provenance."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    source_name: str
    source_uri: str
    strategy: ChunkStrategy
    heading: str | None = None
    page_number: int | None = None
    content_type: str | None = None
    source_headings: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-based limits shared by the MVP chunking strategies."""

    max_chars: int = 1000
    overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")


class Chunker(Protocol):
    """Interface implemented by each interchangeable chunking strategy."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document while preserving its source metadata."""


def document_id_for(document: Document) -> str:
    """Return a stable ID for a source file, shared by its PDF page records."""
    return sha256(document.source_uri.encode("utf-8")).hexdigest()[:24]


def make_chunks(
    document: Document,
    pieces: list[tuple[str, str | None]],
    strategy: ChunkStrategy,
) -> list[Chunk]:
    """Create chunks from (text, heading) pairs using stable content IDs."""
    document_id = document_id_for(document)
    results: list[Chunk] = []
    for index, (raw_text, heading) in enumerate(pieces):
        text = raw_text.strip()
        if not text:
            continue
        identity = "\0".join(
            (
                document_id,
                str(document.page_number or ""),
                str(index),
                text,
            )
        )
        chunk_id = sha256(identity.encode("utf-8")).hexdigest()
        results.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                chunk_index=index,
                source_name=document.source_name,
                source_uri=document.source_uri,
                strategy=strategy,
                heading=heading,
                page_number=document.page_number,
                content_type=document.content_type,
                source_headings=tuple(document.headings),
                metadata=dict(document.metadata),
            )
        )
    return results
