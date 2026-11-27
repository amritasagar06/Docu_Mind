"""Atomic JSONL persistence for the canonical chunk collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from documind.chunking.base import Chunk, ChunkStrategy


class ChunkManifestStore:
    """Store canonical chunks independently from derived search indexes.

    The JSONL manifest is the rebuild source for BM25 and Chroma. It contains
    source-linked chunks only; ranking data is always derived from this file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[Chunk]:
        """Load all canonical chunks, raising an error for malformed rows."""
        if not self.path.exists():
            return []
        chunks: list[Chunk] = []
        seen: set[str] = set()
        with self.path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    chunk = _chunk_from_dict(record)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid chunk manifest row {line_number} in {self.path}: {exc}"
                    ) from exc
                if chunk.chunk_id in seen:
                    raise ValueError(f"Duplicate chunk ID {chunk.chunk_id!r} in {self.path}")
                seen.add(chunk.chunk_id)
                chunks.append(chunk)
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def replace(self, chunks: list[Chunk]) -> int:
        """Atomically replace the manifest with the supplied canonical set."""
        _validate_unique_ids(chunks)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as output:
                for chunk in sorted(chunks, key=lambda item: item.chunk_id):
                    output.write(
                        json.dumps(_chunk_to_dict(chunk), ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return len(chunks)

    def upsert(self, chunks: list[Chunk]) -> int:
        """Add or replace by stable chunk ID and persist the updated manifest."""
        _validate_unique_ids(chunks)
        records = {chunk.chunk_id: chunk for chunk in self.load()}
        records.update({chunk.chunk_id: chunk for chunk in chunks})
        return self.replace(list(records.values()))

    def delete(self, chunk_ids: list[str]) -> list[Chunk]:
        """Remove IDs from the manifest and return the removed canonical chunks."""
        requested = set(chunk_ids)
        current = self.load()
        removed = [chunk for chunk in current if chunk.chunk_id in requested]
        if removed:
            self.replace([chunk for chunk in current if chunk.chunk_id not in requested])
        return removed


def _validate_unique_ids(chunks: list[Chunk]) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("A chunk manifest cannot contain duplicate chunk IDs")


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "source_name": chunk.source_name,
        "source_uri": chunk.source_uri,
        "strategy": chunk.strategy,
        "heading": chunk.heading,
        "page_number": chunk.page_number,
        "content_type": chunk.content_type,
        "source_headings": list(chunk.source_headings),
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(record: dict[str, Any]) -> Chunk:
    if not isinstance(record, dict):
        raise TypeError("each row must be a JSON object")
    required = (
        "chunk_id",
        "document_id",
        "text",
        "chunk_index",
        "source_name",
        "source_uri",
        "strategy",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"missing fields: {', '.join(missing)}")
    strategy = record["strategy"]
    if strategy not in {"fixed", "recursive", "semantic"}:
        raise ValueError(f"unsupported chunk strategy {strategy!r}")
    headings = record.get("source_headings", [])
    metadata = record.get("metadata", {})
    if not isinstance(headings, list) or not all(isinstance(value, str) for value in headings):
        raise TypeError("source_headings must be a list of strings")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON object")
    return Chunk(
        chunk_id=str(record["chunk_id"]),
        document_id=str(record["document_id"]),
        text=str(record["text"]),
        chunk_index=int(record["chunk_index"]),
        source_name=str(record["source_name"]),
        source_uri=str(record["source_uri"]),
        strategy=strategy,  # type: ignore[arg-type]
        heading=record.get("heading"),
        page_number=record.get("page_number"),
        content_type=record.get("content_type"),
        source_headings=tuple(headings),
        metadata=metadata,
    )
