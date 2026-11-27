"""Shared interface for interchangeable text embedding models."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    """Create one stable-dimension vector for each input text."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier used for the vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts as normalized vectors, preserving input order."""
