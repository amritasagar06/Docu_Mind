"""Lazy, CPU-first Sentence Transformers implementation."""

from collections.abc import Sequence
from typing import Any


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SentenceTransformerEmbedder:
    """Create normalized local embeddings with Sentence Transformers.

    The model is loaded only when the first non-empty batch is embedded. That
    keeps package imports and tests from downloading model files unexpectedly.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
        revision: str = DEFAULT_MODEL_REVISION,
        *,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if not device.strip():
            raise ValueError("device cannot be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text in order, rejecting blank input and invalid model output."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every text to embed must be a non-empty string")

        model = self._load_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
        if len(rows) != len(texts):
            raise ValueError("Embedding model returned a different number of vectors")

        result: list[list[float]] = []
        dimension: int | None = None
        for row in rows:
            vector = [float(value) for value in row]
            if not vector:
                raise ValueError("Embedding model returned an empty vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("Embedding model returned inconsistent vector dimensions")
            result.append(vector)
        return result

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Sentence Transformers is missing. Install project dependencies "
                    "with `python -m pip install -e \".[dev]\"`."
                ) from exc
            self._model = SentenceTransformer(
                self._model_name,
                revision=self.revision,
                device=self.device,
            )
        return self._model
