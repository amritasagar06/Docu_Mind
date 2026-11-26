"""Validation and normalization for NVIDIA TechQA-RAG-Eval."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any
import zipfile


DATASET_ID = "nvidia/TechQA-RAG-Eval"
DATASET_REVISION = "0b5bbc84b7f07d6d09d063130e90b716d8d4a32a"
DATASET_URL = "https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval"
DATASET_LICENSE = "Apache-2.0 (as labeled on the Hugging Face dataset card)"
REQUIRED_QUESTION_FIELDS = {"id", "question", "answer", "is_impossible", "contexts"}


class TechQADataError(ValueError):
    """Raised when source data is malformed or fails corpus integrity checks."""


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large source archives fit in memory."""
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_question_rows(rows: Any) -> list[dict[str, Any]]:
    """Validate the dataset's question schema without silently dropping rows."""
    if not isinstance(rows, list):
        raise TechQADataError("train.json must contain a JSON array of question records")

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        location = f"train.json row {index}"
        if not isinstance(row, dict):
            raise TechQADataError(f"{location} must be an object")

        missing = REQUIRED_QUESTION_FIELDS - row.keys()
        if missing:
            raise TechQADataError(f"{location} is missing required fields: {sorted(missing)}")

        question_id = row["id"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise TechQADataError(f"{location} has an empty or invalid id")
        if question_id in seen_ids:
            raise TechQADataError(f"Duplicate question id '{question_id}' at {location}")
        seen_ids.add(question_id)

        if not isinstance(row["question"], str) or not row["question"].strip():
            raise TechQADataError(f"Question '{question_id}' has empty or invalid question text")
        if not isinstance(row["answer"], str):
            raise TechQADataError(f"Question '{question_id}' has a non-text answer")
        if not isinstance(row["is_impossible"], bool):
            raise TechQADataError(f"Question '{question_id}' has a non-boolean is_impossible value")
        if not isinstance(row["contexts"], list):
            raise TechQADataError(f"Question '{question_id}' contexts must be a list")

        for context_index, context in enumerate(row["contexts"]):
            context_location = f"question '{question_id}' context {context_index}"
            if not isinstance(context, dict):
                raise TechQADataError(f"{context_location} must be an object")
            filename = context.get("filename")
            text = context.get("text")
            if not isinstance(filename, str) or not filename.strip():
                raise TechQADataError(f"{context_location} has an empty or invalid filename")
            if not isinstance(text, str):
                raise TechQADataError(f"{context_location} has non-text content")
            if PurePosixPath(filename).name != filename or "\\" in filename:
                raise TechQADataError(f"{context_location} filename must be a basename: {filename!r}")

        validated.append(row)
    return validated


def deterministic_splits(
    question_ids: list[str], seed: int = 42, evaluation_fraction: float = 0.2
) -> dict[str, str]:
    """Assign stable development/evaluation splits independent of input order."""
    if not 0.0 < evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be greater than 0 and less than 1")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question_ids must be unique")

    ordered = sorted(
        question_ids,
        key=lambda question_id: (
            sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest(),
            question_id,
        ),
    )
    evaluation_count = int(len(ordered) * evaluation_fraction + 0.5)
    if len(ordered) > 1:
        evaluation_count = max(1, min(len(ordered) - 1, evaluation_count))
    evaluation_ids = set(ordered[:evaluation_count])
    return {
        question_id: "evaluation" if question_id in evaluation_ids else "development"
        for question_id in question_ids
    }


def _jsonl_write(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _safe_corpus_member(member_name: str) -> str:
    """Return a safe dataset-relative path; reject unexpected archive paths."""
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts or len(member.parts) != 2:
        raise TechQADataError(f"Unexpected corpus archive path: {member_name!r}")
    if member.parts[0] != "corpus" or member.suffix.lower() != ".txt":
        raise TechQADataError(f"Unexpected non-text corpus archive member: {member_name!r}")
    return member.name


def normalize_techqa(
    question_rows: Any,
    corpus_zip_path: str | Path,
    documents_path: str | Path,
    benchmark_path: str | Path,
    *,
    revision: str = DATASET_REVISION,
    split_seed: int = 42,
    evaluation_fraction: float = 0.2,
) -> dict[str, Any]:
    """Normalize the full document corpus and separate benchmark records.

    The raw archive is read in two passes. The first computes content hashes and
    all source/QA references; the second writes one record per unique document
    content. Reference answers and answerability labels are written only to the
    benchmark JSONL, never to the document corpus JSONL.
    """
    rows = validate_question_rows(question_rows)
    archive_path = Path(corpus_zip_path)
    docs_output = Path(documents_path)
    benchmark_output = Path(benchmark_path)
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output.parent.mkdir(parents=True, exist_ok=True)

    question_ids_by_filename: dict[str, set[str]] = defaultdict(set)
    expected_filenames: set[str] = set()
    for row in rows:
        for context in row["contexts"]:
            filename = context["filename"]
            expected_filenames.add(filename)
            question_ids_by_filename[filename].add(row["id"])

    split_by_question = deterministic_splits(
        [row["id"] for row in rows], seed=split_seed, evaluation_fraction=evaluation_fraction
    )

    document_metadata_by_hash: dict[str, dict[str, Any]] = {}
    filename_to_document_id: dict[str, str] = {}
    archive_members: list[tuple[str, str]] = []
    decode_warning_files: list[str] = []
    decode_replacement_characters = 0
    archive_file_count = 0
    archive_total_uncompressed_bytes = 0

    with zipfile.ZipFile(archive_path) as archive:
        file_infos = sorted((info for info in archive.infolist() if not info.is_dir()), key=lambda info: info.filename)

        # Pass 1: hash content and build complete alias/reference maps.
        for info in file_infos:
            filename = _safe_corpus_member(info.filename)
            if filename in filename_to_document_id:
                raise TechQADataError(f"Corpus contains duplicate filename {filename!r}")
            raw_bytes = archive.read(info)
            content_hash = sha256(raw_bytes).hexdigest()
            document_id = f"techqa_doc_{content_hash}"
            try:
                raw_bytes.decode("utf-8-sig", errors="strict")
            except UnicodeDecodeError:
                decoded = raw_bytes.decode("utf-8-sig", errors="replace")
                replacement_count = decoded.count("\ufffd")
                decode_warning_files.append(info.filename)
                decode_replacement_characters += replacement_count
            else:
                replacement_count = 0

            metadata = document_metadata_by_hash.setdefault(
                content_hash,
                {
                    "document_id": document_id,
                    "content_sha256": content_hash,
                    "source_filenames": [],
                    "source_archive_members": [],
                    "source_question_ids": set(),
                    "decode_replacement_characters": 0,
                },
            )
            metadata["source_filenames"].append(filename)
            metadata["source_archive_members"].append(info.filename)
            metadata["decode_replacement_characters"] += replacement_count
            metadata["source_question_ids"].update(question_ids_by_filename.get(filename, set()))
            filename_to_document_id[filename] = document_id
            archive_members.append((info.filename, content_hash))
            archive_file_count += 1
            archive_total_uncompressed_bytes += info.file_size

        missing_contexts = sorted(expected_filenames - filename_to_document_id.keys())
        if missing_contexts:
            raise TechQADataError(
                f"{len(missing_contexts)} benchmark context files are missing from corpus.zip; "
                f"examples: {missing_contexts[:5]}"
            )

        # Stage output so errors do not leave half-written JSONL files.
        docs_fd, docs_temp_name = tempfile.mkstemp(
            prefix=f".{docs_output.name}.", suffix=".tmp", dir=docs_output.parent
        )
        benchmark_fd, benchmark_temp_name = tempfile.mkstemp(
            prefix=f".{benchmark_output.name}.", suffix=".tmp", dir=benchmark_output.parent
        )
        os.close(docs_fd)
        os.close(benchmark_fd)
        try:
            unique_written: set[str] = set()
            with os.fdopen(os.open(docs_temp_name, os.O_WRONLY), "w", encoding="utf-8", newline="\n") as docs_handle:
                for member_name, content_hash in archive_members:
                    if content_hash in unique_written:
                        continue
                    unique_written.add(content_hash)
                    raw_bytes = archive.read(member_name)
                    text = raw_bytes.decode("utf-8-sig", errors="replace")
                    metadata = document_metadata_by_hash[content_hash]
                    source_members = sorted(metadata["source_archive_members"])
                    source_filenames = sorted(metadata["source_filenames"])
                    record = {
                        "document_id": metadata["document_id"],
                        "source_name": source_filenames[0],
                        "source_uri": (
                            f"hf://datasets/{DATASET_ID}@{revision}/"
                            f"corpus.zip!/{source_members[0]}"
                        ),
                        "source_filenames": source_filenames,
                        "source_question_ids": sorted(metadata["source_question_ids"]),
                        "text": text,
                        "headings": [],
                        "page_number": None,
                        "content_type": "text/plain",
                        "content_sha256": content_hash,
                        "dataset_id": DATASET_ID,
                        "dataset_revision": revision,
                        "license": DATASET_LICENSE,
                        "warnings": (
                            ["invalid_utf8_replaced"]
                            if metadata["decode_replacement_characters"]
                            else []
                        ),
                    }
                    _jsonl_write(docs_handle, record)

            benchmark_records = 0
            with os.fdopen(os.open(benchmark_temp_name, os.O_WRONLY), "w", encoding="utf-8", newline="\n") as benchmark_handle:
                for row in rows:
                    source_filenames = sorted({context["filename"] for context in row["contexts"]})
                    expected_document_ids = sorted(
                        {filename_to_document_id[name] for name in source_filenames}
                    )
                    record = {
                        "question_id": row["id"],
                        "question": row["question"],
                        "reference_answer": row["answer"],
                        "is_answerable": not row["is_impossible"],
                        "expected_document_ids": expected_document_ids,
                        "expected_source_filenames": source_filenames,
                        "source_split": "train",
                        "local_split": split_by_question[row["id"]],
                        "dataset_id": DATASET_ID,
                        "dataset_revision": revision,
                    }
                    _jsonl_write(benchmark_handle, record)
                    benchmark_records += 1

            os.replace(docs_temp_name, docs_output)
            os.replace(benchmark_temp_name, benchmark_output)
        except Exception:
            for temp_name in (docs_temp_name, benchmark_temp_name):
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
            raise

    split_counts = {
        name: sum(1 for split in split_by_question.values() if split == name)
        for name in ("development", "evaluation")
    }
    return {
        "question_rows": len(rows),
        "impossible_questions": sum(1 for row in rows if row["is_impossible"]),
        "question_rows_without_contexts": sum(1 for row in rows if not row["contexts"]),
        "unique_context_filenames": len(expected_filenames),
        "corpus_archive_files": archive_file_count,
        "corpus_archive_uncompressed_bytes": archive_total_uncompressed_bytes,
        "unique_document_content": len(document_metadata_by_hash),
        "duplicate_content_files_collapsed": archive_file_count - len(document_metadata_by_hash),
        "unmatched_context_filenames": 0,
        "decode_warning_files": sorted(decode_warning_files),
        "decode_replacement_characters": decode_replacement_characters,
        "benchmark_records": benchmark_records,
        "local_split_seed": split_seed,
        "local_evaluation_fraction": evaluation_fraction,
        "local_split_counts": split_counts,
        "documents_output": str(docs_output),
        "benchmark_output": str(benchmark_output),
    }
