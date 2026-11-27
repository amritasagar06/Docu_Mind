"""Compare dense, sparse, and RRF hybrid retrieval on held-out TechQA cases.

Only question text is supplied to search. Expected document IDs and labels are
used after retrieval to compute metrics, and reference answers are never read.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import platform
from pathlib import Path
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documind.embeddings.sentence_transformers import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbedder,
)
from documind.retrieval import LocalIndexManager  # noqa: E402


BENCHMARK = ROOT / "data/processed/techqa_benchmark_cases.jsonl"
SOURCE_MANIFEST = ROOT / "data/manifests/techqa.json"
INDEX_DIR = ROOT / "data/indexes/techqa"
RESULTS_DIR = ROOT / "data/eval/results"
MODES = ("dense", "sparse", "hybrid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3, help="development questions per mode for warm-up")
    args = parser.parse_args()
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    for required in (BENCHMARK, SOURCE_MANIFEST, INDEX_DIR / "index_config.json"):
        if not required.is_file():
            raise SystemExit(f"Missing {required}; run the documented TechQA preparation first")

    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    index_config = json.loads((INDEX_DIR / "index_config.json").read_text(encoding="utf-8"))
    if index_config["embedding_model"] != DEFAULT_MODEL_NAME or index_config["embedding_revision"] != DEFAULT_MODEL_REVISION:
        raise SystemExit("Index model configuration does not match this evaluation runner")
    if index_config["dataset_revision"] != source_manifest["revision"]:
        raise SystemExit("Index and dataset manifest revisions do not match")

    rows = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluation = [row for row in rows if row["local_split"] == "evaluation"]
    development = [row for row in rows if row["local_split"] == "development"]
    eligible = [row for row in evaluation if row["is_answerable"] and row["expected_document_ids"]]
    no_target = [row for row in evaluation if row["is_answerable"] and not row["expected_document_ids"]]
    unanswerable = [row for row in evaluation if not row["is_answerable"]]
    if not eligible:
        raise SystemExit("No answerable evaluation cases with expected document IDs")

    manager = LocalIndexManager(
        INDEX_DIR,
        SentenceTransformerEmbedder(device="cpu"),
        collection_name="techqa_chunks",
        rrf_constant=60,
    )
    if manager.vector_store.count != manager.keyword_index.count or not manager.vector_store.count:
        raise SystemExit("Dense and sparse indexes are missing or inconsistent; run scripts/index_techqa.py")

    # Warm only with development questions. No labels or answers enter search.
    warmups = development[: args.warmup]
    for row in warmups:
        for mode in MODES:
            manager.search(row["question"], mode=mode, top_k=10, dense_k=10, sparse_k=10)

    timings: dict[str, list[float]] = {mode: [] for mode in MODES}
    records: list[dict[str, object]] = []
    for case_index, row in enumerate(eligible):
        expected = set(row["expected_document_ids"])
        # Rotate which algorithm runs first to reduce systematic order effects.
        order = MODES[case_index % len(MODES) :] + MODES[: case_index % len(MODES)]
        results_by_mode = {}
        for mode in order:
            started = perf_counter()
            results = manager.search(
                row["question"], mode=mode, top_k=10, dense_k=10, sparse_k=10
            )
            elapsed_ms = (perf_counter() - started) * 1000
            timings[mode].append(elapsed_ms)
            results_by_mode[mode] = results
        for mode in MODES:
            results = results_by_mode[mode]
            ranked_doc_ids = [result.chunk.document_id for result in results]
            relevant_ranks = [rank for rank, document_id in enumerate(ranked_doc_ids, start=1) if document_id in expected]
            hit_docs_5 = set(ranked_doc_ids[:5]) & expected
            hit_docs_10 = set(ranked_doc_ids[:10]) & expected
            records.append({
                "question_id": row["question_id"],
                "mode": mode,
                "expected_document_count": len(expected),
                "retrieved_document_ids_at_10": ranked_doc_ids,
                "recall_at_5": len(set(ranked_doc_ids[:5]) & expected) / len(expected),
                "recall_at_10": len(hit_docs_10) / len(expected),
                "reciprocal_rank_at_10": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
                "latency_ms": timings[mode][-1],
            })
        if (case_index + 1) % 10 == 0 or case_index + 1 == len(eligible):
            print(f"Evaluated {case_index + 1}/{len(eligible)} eligible held-out questions", flush=True)

    summary = {
        "dataset_id": source_manifest["dataset_id"],
        "dataset_revision": source_manifest["revision"],
        "benchmark_sha256": _sha256(BENCHMARK),
        "normalized_documents_sha256": source_manifest["normalized_files"]["techqa_documents.jsonl"]["sha256"],
        "evaluation_split": "evaluation",
        "evaluation_split_seed": source_manifest["normalization"]["stats"]["local_split_seed"],
        "evaluated_answerable_with_targets": len(eligible),
        "excluded_unanswerable": len(unanswerable),
        "excluded_answerable_without_targets": len(no_target),
        "evaluation_cases_total": len(evaluation),
        "warmup_development_questions_per_mode": len(warmups),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "chromadb": metadata.version("chromadb"),
            "sentence_transformers": metadata.version("sentence-transformers"),
            "rank_bm25": metadata.version("rank-bm25"),
        },
        "indexed_chunk_count": manager.vector_store.count,
        "indexed_document_count": len({chunk.document_id for chunk in manager.chunks()}),
        "retrieval_configuration": {
            "chunk_strategy": index_config["chunking"],
            "embedding_model": index_config["embedding_model"],
            "embedding_revision": index_config["embedding_revision"],
            "device": "cpu",
            "dense_k": 10,
            "sparse_k": 10,
            "returned_top_k": 10,
            "rrf_constant": 60,
        },
        "metrics_definition": {
            "recall_at_k": "Macro mean across eligible questions of distinct expected documents found among the top k returned chunks divided by expected document count.",
            "mrr_at_10": "Macro mean reciprocal rank of the first chunk whose canonical document ID is expected; zero when none appears in top 10.",
            "latency": "Retrieval call wall time per question; excludes model loading via development warm-up, index build, and evaluation I/O.",
        },
        "results": {},
        "limitations": [
            "TechQA evaluation questions and expected context documents originate from the same source corpus; this measures recovery of known evidence, not generalization to unseen organizations or documents.",
            "Unanswerable and answerable-without-target cases are excluded from recall and MRR because no expected document set exists; this report does not measure abstention quality.",
            "Chunk ranking is mapped to document IDs; retrieving several chunks from one document counts once toward recall.",
            "Latency is local CPU measurements and depends on hardware, cache state, and background load; hybrid performs both dense and BM25 retrieval.",
            "This compares retrieval only; it does not measure answer correctness, faithfulness, or citation quality.",
        ],
    }
    for mode in MODES:
        mode_rows = [record for record in records if record["mode"] == mode]
        latency = timings[mode]
        summary["results"][mode] = {
            "n": len(mode_rows),
            "recall_at_5": statistics.fmean(record["recall_at_5"] for record in mode_rows),
            "recall_at_10": statistics.fmean(record["recall_at_10"] for record in mode_rows),
            "mrr_at_10": statistics.fmean(record["reciprocal_rank_at_10"] for record in mode_rows),
            "latency_ms_mean": statistics.fmean(latency),
            "latency_ms_median": statistics.median(latency),
            "latency_ms_p95": _percentile(latency, 0.95),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records_path = RESULTS_DIR / "techqa_retrieval_cases.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in sorted(records, key=lambda item: (str(item["question_id"]), str(item["mode"]))):
            output.write(json.dumps(record, sort_keys=True) + "\n")
    summary["per_case_results"] = str(records_path.relative_to(ROOT)).replace("\\", "/")
    (RESULTS_DIR / "techqa_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
