"""Validate and normalize the pinned TechQA corpus and benchmark records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from documind.datasets.techqa import (  # noqa: E402
    DATASET_ID,
    DATASET_REVISION,
    TechQADataError,
    normalize_techqa,
    sha256_file,
)


RAW_DIR = ROOT / "data" / "raw" / "techqa"
MANIFEST_PATH = ROOT / "data" / "manifests" / "techqa.json"
PROCESSED_DIR = ROOT / "data" / "processed"


def _verify_source_manifest(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    if manifest.get("dataset_id") != DATASET_ID:
        raise TechQADataError(f"Manifest dataset_id must be {DATASET_ID!r}")
    if manifest.get("revision") != DATASET_REVISION:
        raise TechQADataError(
            f"Manifest revision must be the pinned revision {DATASET_REVISION}; got {manifest.get('revision')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise TechQADataError("Manifest is missing its source file inventory")

    for filename in ("train.json", "corpus.zip"):
        record = files.get(filename)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise TechQADataError(f"Manifest is missing checksum for {filename}")
        source_path = RAW_DIR / filename
        if not source_path.is_file():
            raise TechQADataError(f"Missing raw file {source_path}; run scripts/download_techqa.py first")
        actual_hash = sha256_file(source_path)
        if actual_hash != record["sha256"]:
            raise TechQADataError(
                f"Checksum mismatch for {filename}: expected {record['sha256']}, got {actual_hash}"
            )
        if source_path.stat().st_size != record.get("size_bytes"):
            raise TechQADataError(f"File size in manifest does not match {filename}")
    return files


def _atomic_json_write(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="seed for the deterministic local split")
    parser.add_argument(
        "--evaluation-fraction",
        type=float,
        default=0.2,
        help="fraction of benchmark questions assigned to the evaluation partition",
    )
    args = parser.parse_args()

    if not MANIFEST_PATH.is_file():
        raise SystemExit("Dataset manifest missing; run python scripts/download_techqa.py first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    _verify_source_manifest(manifest)
    question_rows = json.loads((RAW_DIR / "train.json").read_text(encoding="utf-8"))

    stats = normalize_techqa(
        question_rows,
        RAW_DIR / "corpus.zip",
        PROCESSED_DIR / "techqa_documents.jsonl",
        PROCESSED_DIR / "techqa_benchmark_cases.jsonl",
        revision=DATASET_REVISION,
        split_seed=args.seed,
        evaluation_fraction=args.evaluation_fraction,
    )
    stats["documents_output"] = "data/processed/techqa_documents.jsonl"
    stats["benchmark_output"] = "data/processed/techqa_benchmark_cases.jsonl"

    normalized_files: dict[str, object] = {}
    for filename in ("techqa_documents.jsonl", "techqa_benchmark_cases.jsonl"):
        path = PROCESSED_DIR / filename
        normalized_files[filename] = {
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest["normalization"] = {
        "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "content-addressed documents; SHA-256-ranked deterministic question split",
        "stats": stats,
    }
    manifest["normalized_files"] = normalized_files
    _atomic_json_write(MANIFEST_PATH, manifest)

    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Updated manifest: {MANIFEST_PATH.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
