"""Canonical record returned by document loaders."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Document:
    """An extracted document or page, with source and structural metadata."""

    text: str
    source_name: str
    source_uri: str
    headings: list[str] = field(default_factory=list)
    page_number: int | None = None
    content_type: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path, text: str, **kwargs: object) -> "Document":
        """Build a document with consistent local-file source fields."""
        return cls(
            text=text,
            source_name=path.name,
            source_uri=path.resolve().as_uri(),
            **kwargs,
        )
