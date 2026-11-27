"""Run a small CPU embedding + persistent Chroma smoke test.

This downloads the pinned Sentence Transformers model on its first run and writes
one small test collection under the Git-ignored ``data/indexes/`` directory.
"""

from pathlib import Path
import subprocess
import sys

from documind.chunking import FixedSizeChunker
from documind.embeddings import SentenceTransformerEmbedder
from documind.ingestion.models import Document
from documind.vectorstore import ChromaVectorStore


def main(*, verify_existing: bool = False) -> None:
    document = Document(
        text=(
            "# Database connection pool\n\n"
            "Set MAX_POOL_SIZE in the database configuration to control the "
            "maximum number of reusable connections."
        ),
        source_name="day4-smoke-test.md",
        source_uri="documind://smoke-test/day4/database-guide",
        headings=["Database connection pool"],
        content_type="text/markdown",
        metadata={"purpose": "day4-smoke-test"},
    )
    chunks = FixedSizeChunker().chunk(document)
    embedder = SentenceTransformerEmbedder(device="cpu")
    index_path = Path("data/indexes/day4-smoke")

    store = ChromaVectorStore(index_path, embedder, collection_name="day4_smoke")
    if not verify_existing:
        store.rebuild(chunks)
    elif store.count != len(chunks):
        raise SystemExit("Smoke test failed: persisted chunk count does not match")
    results = store.query("How can I set the maximum reusable database connections?", 3)
    if not results or results[0].chunk.chunk_id != chunks[0].chunk_id:
        raise SystemExit("Smoke test failed: query did not return the indexed source chunk")

    print("Day 4 smoke test passed (CPU embeddings and Chroma query).")
    print(f"Top source: {results[0].chunk.source_name}")
    print(f"Similarity: {results[0].score:.4f}")
    print(f"Index: {index_path.resolve()}")
    if not verify_existing:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--verify-existing"],
            check=True,
        )


if __name__ == "__main__":
    main(verify_existing="--verify-existing" in sys.argv[1:])
