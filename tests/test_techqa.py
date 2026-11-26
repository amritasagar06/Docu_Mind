import json
from pathlib import Path
import zipfile

import pytest

from documind.datasets.techqa import (
    DATASET_REVISION,
    TechQADataError,
    deterministic_splits,
    normalize_techqa,
    validate_question_rows,
)


def write_corpus(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in entries.items():
            archive.writestr(f"corpus/{name}", text)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def valid_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "Q1",
            "question": "How do I configure the endpoint?",
            "answer": "REFERENCE_ANSWER_MUST_NOT_ENTER_CORPUS_Q1",
            "is_impossible": False,
            "contexts": [{"filename": "guide-a.txt", "text": "Context A"}],
        },
        {
            "id": "Q2",
            "question": "How do I set the endpoint?",
            "answer": "REFERENCE_ANSWER_MUST_NOT_ENTER_CORPUS_Q2",
            "is_impossible": False,
            "contexts": [{"filename": "guide-b.txt", "text": "Context B"}],
        },
        {
            "id": "Q3",
            "question": "What is the secret phone number?",
            "answer": "-",
            "is_impossible": True,
            "contexts": [],
        },
    ]


def test_validation_rejects_missing_fields_and_duplicate_ids() -> None:
    with pytest.raises(TechQADataError, match="missing required fields"):
        validate_question_rows([{"id": "Q1"}])

    duplicate = valid_rows()
    duplicate[1]["id"] = "Q1"
    with pytest.raises(TechQADataError, match="Duplicate question id"):
        validate_question_rows(duplicate)


def test_deterministic_split_is_stable_and_order_independent() -> None:
    ids = [f"Q{i}" for i in range(10)]

    first = deterministic_splits(ids, seed=7, evaluation_fraction=0.2)
    second = deterministic_splits(list(reversed(ids)), seed=7, evaluation_fraction=0.2)

    assert first == second
    assert list(first.values()).count("evaluation") == 2


def test_normalization_deduplicates_documents_and_keeps_benchmark_separate(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.zip"
    write_corpus(
        archive_path,
        {
            "guide-a.txt": b"Shared technical guidance.",
            "guide-b.txt": b"Shared technical guidance.",
            "other.txt": b"Different reference document.",
        },
    )
    document_path = tmp_path / "processed" / "documents.jsonl"
    benchmark_path = tmp_path / "processed" / "benchmark.jsonl"

    stats = normalize_techqa(
        valid_rows(),
        archive_path,
        document_path,
        benchmark_path,
        revision=DATASET_REVISION,
        split_seed=9,
        evaluation_fraction=0.34,
    )
    documents = read_jsonl(document_path)
    benchmark = read_jsonl(benchmark_path)

    assert len(documents) == 2
    shared = next(document for document in documents if document["text"] == "Shared technical guidance.")
    assert shared["source_filenames"] == ["guide-a.txt", "guide-b.txt"]
    assert shared["source_question_ids"] == ["Q1", "Q2"]
    assert shared["document_id"].startswith("techqa_doc_")
    assert all("reference_answer" not in document for document in documents)
    assert "REFERENCE_ANSWER_MUST_NOT_ENTER_CORPUS_Q1" not in document_path.read_text(encoding="utf-8")

    by_id = {record["question_id"]: record for record in benchmark}
    assert by_id["Q1"]["expected_document_ids"] == [shared["document_id"]]
    assert by_id["Q3"]["is_answerable"] is False
    assert by_id["Q3"]["expected_document_ids"] == []
    assert by_id["Q1"]["reference_answer"] == "REFERENCE_ANSWER_MUST_NOT_ENTER_CORPUS_Q1"
    assert set(record["local_split"] for record in benchmark) <= {"development", "evaluation"}
    assert stats["question_rows"] == 3
    assert stats["impossible_questions"] == 1
    assert stats["duplicate_content_files_collapsed"] == 1


def test_normalization_outputs_are_byte_for_byte_deterministic(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.zip"
    write_corpus(archive_path, {"guide-a.txt": b"Stable document text."})
    outputs = [
        (tmp_path / f"documents-{index}.jsonl", tmp_path / f"benchmark-{index}.jsonl")
        for index in (1, 2)
    ]

    for document_path, benchmark_path in outputs:
        normalize_techqa(
            [valid_rows()[0]],
            archive_path,
            document_path,
            benchmark_path,
            split_seed=42,
            evaluation_fraction=0.2,
        )

    assert outputs[0][0].read_bytes() == outputs[1][0].read_bytes()
    assert outputs[0][1].read_bytes() == outputs[1][1].read_bytes()


def test_missing_expected_context_is_reported_and_fails(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.zip"
    write_corpus(archive_path, {"different.txt": b"Not the linked source."})

    with pytest.raises(TechQADataError, match="benchmark context files are missing"):
        normalize_techqa(
            [valid_rows()[0]],
            archive_path,
            tmp_path / "documents.jsonl",
            tmp_path / "benchmark.jsonl",
        )


def test_invalid_utf8_is_retained_with_a_warning(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.zip"
    write_corpus(archive_path, {"bad-encoding.txt": b"Text with invalid byte: \xff"})
    document_path = tmp_path / "documents.jsonl"

    stats = normalize_techqa(
        [], archive_path, document_path, tmp_path / "benchmark.jsonl"
    )
    [document] = read_jsonl(document_path)

    assert "\ufffd" in document["text"]
    assert document["warnings"] == ["invalid_utf8_replaced"]
    assert stats["decode_warning_files"] == ["corpus/bad-encoding.txt"]
    assert stats["decode_replacement_characters"] == 1


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(TechQADataError, match="Unexpected corpus archive path"):
        normalize_techqa(
            [], archive_path, tmp_path / "documents.jsonl", tmp_path / "benchmark.jsonl"
        )


def test_context_paths_must_be_basenames() -> None:
    row = valid_rows()[0]
    row["contexts"] = [{"filename": "../guide-a.txt", "text": "Context A"}]

    with pytest.raises(TechQADataError, match="filename must be a basename"):
        validate_question_rows([row])
