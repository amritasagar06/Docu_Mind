"""Markdown-aware sectioning and recursive boundary-based splitting."""

import re

from documind.chunking.base import Chunk, ChunkingConfig, make_chunks
from documind.ingestion.models import Document


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def markdown_sections(text: str) -> list[tuple[str, str | None]]:
    """Return section bodies paired with their Markdown heading path."""
    sections: list[tuple[str, str | None]] = []
    heading_stack: list[tuple[int, str]] = []
    body: list[str] = []
    current_heading: str | None = None

    def flush() -> None:
        section_text = "\n".join(body).strip()
        if section_text:
            sections.append((section_text, current_heading))

    for line in text.splitlines():
        match = _MARKDOWN_HEADING.match(line)
        if not match:
            body.append(line)
            continue

        flush()
        body.clear()
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
        current_heading = " > ".join(item[1] for item in heading_stack)

    flush()
    return sections


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split an unbroken token only when no natural boundary remains."""
    return [text[start : start + max_chars].strip() for start in range(0, len(text), max_chars) if text[start : start + max_chars].strip()]


def _recursive_split(text: str, max_chars: int, separator_index: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    if separator_index >= len(_SEPARATORS):
        return _hard_split(text, max_chars)

    separator = _SEPARATORS[separator_index]
    if separator not in text:
        return _recursive_split(text, max_chars, separator_index + 1)

    parts = text.split(separator)
    output: list[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_chars:
            if current:
                output.append(current)
                current = ""
            output.extend(_recursive_split(part, max_chars, separator_index + 1))
            continue

        candidate = f"{current}{separator if current else ''}{part}"
        if current and len(candidate) > max_chars:
            output.append(current)
            current = part
        else:
            current = candidate

    if current:
        output.append(current)
    return output


class RecursiveChunker:
    """Split by Markdown sections, then paragraphs, lines, sentences, and words."""

    strategy = "recursive"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []

        pieces: list[tuple[str, str | None]] = []
        for section_text, heading in markdown_sections(document.text):
            pieces.extend(
                (text, heading)
                for text in _recursive_split(section_text, self.config.max_chars)
            )
        return make_chunks(document, pieces, self.strategy)
