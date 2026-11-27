"""Build the pinned local TechQA corpus index used by offline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.chunking import ChunkingConfig, FixedSizeChunker  # noqa: E402
from documind.embeddings.sentence_transformers import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbedder,
)
from documind.ingestion.models import Document  # noqa: E402
from documind.retrieval import LocalIndexManager  # noqa: E402


DOCS_PATH = ROOT / "data/processed/techqa_documents.jsonl"
SOURCE_MANIFEST = ROOT / "data/manifests/techqa.json"
INDEX_DIR = ROOT / "data/indexes/techqa"
CONFIG_PATH = INDEX_DIR / "index_config.json"
MAX_CHARS = 1000
OVERLAP_CHARS = 150
EMBED_BATCH_SIZE = 128
UPSERT_BATCH_SIZE = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_chunks() -> list:
    chunker = FixedSizeChunker(ChunkingConfig(MAX_CHARS, OVERLAP_CHARS))
    chunks = []
    with DOCS_PATH.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            document = Document(
                text=record["text"],
                source_name=record["source_name"],
                source_uri=record["source_uri"],
                headings=record.get("headings", []),
                page_number=record.get("page_number"),
                content_type=record.get("content_type"),
                metadata={
                    "dataset_id": record["dataset_id"],
                    "dataset_revision": record["dataset_revision"],
                    "content_sha256": record["content_sha256"],
                    "license": record["license"],
                    "source_filenames": ",".join(record["source_filenames"]),
                },
            )
            document_chunks = chunker.chunk(document)
            # The benchmark expects these content-addressed canonical IDs.
            for chunk in document_chunks:
                chunk.document_id = record["document_id"]
            chunks.extend(document_chunks)
            if line_number % 2000 == 0:
                print(f"Chunked {line_number:,} documents; {len(chunks):,} chunks", flush=True)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild the index even when its config matches"
    )
    args = parser.parse_args()
    if not DOCS_PATH.is_file() or not SOURCE_MANIFEST.is_file():
        raise SystemExit("TechQA normalized data is missing; run scripts/normalize_techqa.py first")

    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    docs_hash = _sha256(DOCS_PATH)
    config = {
        "dataset_id": source_manifest["dataset_id"],
        "dataset_revision": source_manifest["revision"],
        "normalized_documents_sha256": docs_hash,
        "document_count": source_manifest["normalization"]["stats"]["unique_document_content"],
        "chunking": {"strategy": "fixed", "max_chars": MAX_CHARS, "overlap_chars": OVERLAP_CHARS},
        "embedding_model": DEFAULT_MODEL_NAME,
        "embedding_revision": DEFAULT_MODEL_REVISION,
        "embedding_device": "cpu",
        "embedding_batch_size": EMBED_BATCH_SIZE,
        "vector_upsert_batch_size": UPSERT_BATCH_SIZE,
        "rrf_constant": 60,
        "dense_k": 10,
        "sparse_k": 10,
    }
    if CONFIG_PATH.is_file() and not args.force:
        existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if existing == config:
            manager = LocalIndexManager(
                INDEX_DIR,
                SentenceTransformerEmbedder(device="cpu", batch_size=EMBED_BATCH_SIZE),
                collection_name="techqa_chunks",
                rrf_constant=60,
            )
            if manager.vector_store.count == manager.keyword_index.count:
                print(f"Index already matches config: {manager.vector_store.count:,} chunks")
                return 0
        else:
            raise SystemExit("Existing index config differs; inspect it and rerun with --force to rebuild")

    chunks = _load_chunks()
    if not chunks:
        raise SystemExit("No chunks were generated from TechQA documents")
    print(f"Building dense and sparse indexes for {len(chunks):,} chunks", flush=True)
    manager = LocalIndexManager(
        INDEX_DIR,
        SentenceTransformerEmbedder(device="cpu", batch_size=EMBED_BATCH_SIZE),
        collection_name="techqa_chunks",
        rrf_constant=60,
    )
    manager.rebuild(chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Indexed {manager.vector_store.count:,} chunks at {INDEX_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
