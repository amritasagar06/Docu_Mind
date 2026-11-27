"""Sparse retrieval and local index coordination."""

from documind.retrieval.bm25 import BM25Result, BM25SearchIndex, tokenize
from documind.retrieval.duplicates import (
    DuplicateCandidate,
    DuplicateDetector,
    DuplicateRegistry,
    DuplicateStatus,
)
from documind.retrieval.index import LocalIndexManager
from documind.retrieval.hybrid import (
    HybridRetriever,
    ReciprocalRankFusion,
    RetrievalMode,
    RetrievalResult,
)
from documind.retrieval.manifest import ChunkManifestStore
from documind.retrieval.reranking import (
    CrossEncoderReranker,
    IdentityReranker,
    RerankRun,
    RerankedResult,
    RerankingPipeline,
    Reranker,
)

__all__ = [
    "BM25Result",
    "BM25SearchIndex",
    "ChunkManifestStore",
    "CrossEncoderReranker",
    "DuplicateCandidate",
    "DuplicateDetector",
    "DuplicateRegistry",
    "DuplicateStatus",
    "HybridRetriever",
    "IdentityReranker",
    "LocalIndexManager",
    "ReciprocalRankFusion",
    "RetrievalMode",
    "RetrievalResult",
    "RerankRun",
    "RerankedResult",
    "RerankingPipeline",
    "Reranker",
    "tokenize",
]
