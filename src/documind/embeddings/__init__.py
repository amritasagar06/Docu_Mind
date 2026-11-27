"""Local text embedding interfaces and implementations."""

from documind.embeddings.base import EmbeddingModel
from documind.embeddings.sentence_transformers import SentenceTransformerEmbedder

__all__ = ["EmbeddingModel", "SentenceTransformerEmbedder"]
