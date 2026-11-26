"""Fixed-size character chunking with configurable overlap."""

from documind.chunking.base import Chunk, ChunkingConfig, make_chunks
from documind.ingestion.models import Document


class FixedSizeChunker:
    """Split text into bounded windows, preferring whitespace boundaries."""

    strategy = "fixed"

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text.strip():
            return []

        spans: list[tuple[str, str | None]] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.config.max_chars, length)
            if end < length:
                minimum_boundary = start + max(1, self.config.max_chars // 2)
                whitespace = max(text.rfind(" ", minimum_boundary, end), text.rfind("\n", minimum_boundary, end), text.rfind("\t", minimum_boundary, end))
                if whitespace > start:
                    end = whitespace

            piece = text[start:end].strip()
            if piece:
                spans.append((piece, None))
            if end >= length:
                break

            next_start = end - self.config.overlap_chars
            if next_start <= start:
                next_start = end
            while next_start < length and text[next_start].isspace():
                next_start += 1
            start = next_start

        return make_chunks(document, spans, self.strategy)
