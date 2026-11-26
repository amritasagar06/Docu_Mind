"""Semantic grouping using an injected text-similarity function."""

from collections.abc import Callable
import re

from documind.chunking.base import Chunk, ChunkingConfig, make_chunks
from documind.chunking.recursive import _recursive_split, markdown_sections
from documind.ingestion.models import Document


SimilarityFunction = Callable[[str, str], float]


def _paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs or ([text.strip()] if text.strip() else [])


class SemanticChunker:
    """Group adjacent paragraphs while their semantic similarity stays high.

    Inject `similarity_fn` so the algorithm can be tested without a model and
    later connected to the project's local embedding implementation.
    """

    strategy = "semantic"

    def __init__(
        self,
        similarity_fn: SimilarityFunction,
        config: ChunkingConfig | None = None,
        similarity_threshold: float = 0.72,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.similarity_fn = similarity_fn
        self.config = config or ChunkingConfig()
        self.similarity_threshold = similarity_threshold

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.text.strip():
            return []

        pieces: list[tuple[str, str | None]] = []
        for section_text, heading in markdown_sections(document.text):
            units: list[str] = []
            for paragraph in _paragraphs(section_text):
                units.extend(_recursive_split(paragraph, self.config.max_chars))

            current = ""
            for unit in units:
                candidate = f"{current}\n\n{unit}" if current else unit
                can_fit = len(candidate) <= self.config.max_chars
                is_similar = bool(current) and self.similarity_fn(current, unit) >= self.similarity_threshold
                if current and (not can_fit or not is_similar):
                    pieces.append((current, heading))
                    current = unit
                else:
                    current = candidate
            if current:
                pieces.append((current, heading))

        return make_chunks(document, pieces, self.strategy)
