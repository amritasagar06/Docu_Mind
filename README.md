# Docu_Mind — DocuMind AI

> **Find answers in technical documentation, with the evidence attached.**

DocuMind AI is an early-stage, local-first documentation assistant for engineering and IT teams. It is being built to make technical knowledge easier to search and answers easier to verify. The project is under active development: Days 1–7 provide document ingestion, chunking, dataset preparation, local embeddings, vector search, keyword search, reviewable duplicate flags, hybrid ranking, and a traceable reranking interface. A chat interface and generated answers are planned, not implemented yet.

## The problem

Technical instructions are scattered across PDFs, Markdown files, support articles, runbooks, and archived Q&A. Keyword search can miss questions phrased differently from the source. Vector search can miss exact identifiers such as `MAX_POOL_SIZE` or `HTTP 422`. And a language model can give a confident response that the documentation never supported.

DocuMind AI is being built around a simple product promise: make the source passages easy to find and inspect, and only add answer generation when it can point back to evidence.

## What DocuMind can do today

- Load Markdown, text, HTML, and PDF files into a consistent document format.
- Preserve headings and source details; keep PDF pages separate with page numbers.
- Split documents into fixed-size, Markdown-aware recursive, or semantic chunks.
- Acquire and normalize the pinned TechQA-RAG-Eval data into document records and separate benchmark cases.
- Create local, CPU-based embeddings with a pinned Sentence Transformers model revision.
- Store chunks in a persistent ChromaDB collection; add, search, delete, and rebuild chunks using stable IDs.
- Search exact identifiers and error codes with BM25 keyword retrieval.
- Search in dense-only, sparse-only, or hybrid mode; hybrid mode combines ranked lists with configurable Reciprocal Rank Fusion (RRF).
- Rerank a bounded hybrid candidate list with a no-model identity baseline or an explicitly configured local cross-encoder; inspect both original retrieval and reranking scores.
- Keep a canonical JSONL chunk manifest as the source for both vector and keyword indexes.
- Flag likely duplicate chunks using a configurable cosine threshold; record review decisions without deleting source chunks.
- Return the matching source chunk and its metadata for inspection.

**Current boundary:** This repository does not yet provide a graphical interface, chat assistant, or generated answers. Retrieval returns source passages; it does not compose or verify a natural-language answer yet.

## Who it is for

The initial product is aimed at software engineers, IT support, and internal technical teams who need to find information in technical documentation. The current build is a local prototype for public or non-confidential material. It is not a production-ready, multi-tenant company knowledge service.

## How it works

```text
Implemented: document, dataset, and local retrieval foundations (Days 1–6)

Public documents ──> format loaders ──> normalized documents
                                            │
                                            ▼
                                   chunking strategies
                                            │
                                            ▼
                               canonical JSONL manifest
                                    ┌───────┴────────┐
                                    ▼                ▼
                          local embeddings         BM25
                                    ▼                │
                             persistent ChromaDB    │
                                    └────────┬───────┘
                                             ▼
                                      source passages
                                      + duplicate review
                                            │
Question ──> dense + BM25 results ──> RRF ──> ranked source chunks

TechQA-RAG-Eval ──> pinned download + checksums ──> normalization
                                                       ├─> seed documents
                                                       └─> separate benchmark cases

Planned product workflow

Question ──> optional reranking ──> grounded answer with citations or abstention
         ──> API + dashboard; evaluation and usage analytics
```

## Dataset

The first real-world dataset is NVIDIA’s [`TechQA-RAG-Eval`](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval), hosted on Hugging Face. It contains technical-support questions from IBM developer forums and associated technical documents. We use it both as a seed document corpus and as a separate offline retrieval benchmark.

The downloaded snapshot is pinned to a specific dataset revision and verified with SHA-256 checksums. The observed snapshot has 910 question rows, including 300 labeled unanswerable, and 28,478 unique document contents after exact-content deduplication. The local question partition contains 728 development and 182 evaluation cases. These values describe the acquired snapshot, not claims about general product accuracy.

Benchmark answers and answerability labels are kept separate from indexed document text. The dataset has been downloaded and normalized, but the complete TechQA document set has not yet been chunked and loaded into the shared search indexes. Raw and processed data are not included in Git by default. See [DATA_SOURCES.md](DATA_SOURCES.md), [PROJECT_GUIDE.md](PROJECT_GUIDE.md), and the tracked [dataset manifest](data/manifests/techqa.json) for provenance and implementation details.

## Technology

| Area | Technology | Status |
| --- | --- | --- |
| Runtime and packaging | Python 3.11+, `pyproject.toml` | Implemented |
| Document loading | BeautifulSoup, PyMuPDF | Day 1 |
| Chunking | Fixed-size, recursive, semantic | Day 2 |
| Dataset pipeline | Hugging Face TechQA-RAG-Eval, pinned snapshot | Day 3 |
| Local embeddings | Sentence Transformers, `all-MiniLM-L6-v2` at a pinned revision | Day 4 |
| Persistent vector store | ChromaDB | Day 4 |
| Keyword retrieval | `rank-bm25` | Day 5 |
| Duplicate review | Chroma cosine candidates + reversible JSONL registry | Day 5 |
| Hybrid ranking | In-house Reciprocal Rank Fusion (default constant 60) | Day 6 |
| Answer generation | Ollama with a locally runnable model | Planned |
| API and dashboard | FastAPI and Streamlit | Planned |
| Metadata and logs | SQLite | Planned |
| Tests | pytest | In use |

## Get started

### Requirements

- Python 3.11 or newer
- `pip`
- Internet access for the optional TechQA download and for the first real embedding-model run

### Windows PowerShell

From the repository folder:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If using another installed Python version, replace `3.12` with that version. On macOS or Linux, activate with `source .venv/bin/activate`.

### Run the tests

```powershell
python -m pytest -q
```

The test suite uses temporary files and a deterministic fake embedder, so it does not download the TechQA dataset or embedding weights.

### Verify real local embeddings and persistence

```powershell
python scripts\smoke_test_vectorstore.py
```

The first run downloads the pinned `all-MiniLM-L6-v2` model revision from Hugging Face (about 91 MB of model weights). It runs on CPU by default, indexes a small sample, performs a query, and launches a fresh process to verify that the index persisted. The sample collection is written under `data/indexes/day4-smoke`, which Git ignores.

### Download and prepare TechQA

```powershell
python scripts\download_techqa.py
python scripts\normalize_techqa.py
```

The download is roughly 50 MB. It is not required for the unit tests or the smoke test. The scripts verify the pinned source and write normalized documents and benchmark records under ignored `data/` paths.

### Use the vector store in Python

```python
from pathlib import Path

from documind.chunking import FixedSizeChunker
from documind.embeddings import SentenceTransformerEmbedder
from documind.ingestion import load_document
from documind.retrieval import IdentityReranker, LocalIndexManager, RerankingPipeline

document = load_document(Path("docs/guide.md"))[0]
chunks = FixedSizeChunker().chunk(document)

index = LocalIndexManager(
    "data/indexes/documind",
    SentenceTransformerEmbedder(device="cpu"),
)
index.add(chunks)

for result in index.vector_store.query("How do I configure the connection pool?", n_results=5):
    print(result.score, result.chunk.source_name, result.chunk.text)

for result in index.keyword_index.query("MAX_POOL_SIZE", n_results=5):
    print(result.score, result.chunk.source_name, result.chunk.text)

for result in index.search(
    "How do I change MAX_POOL_SIZE?",
    mode="hybrid",
    top_k=5,
    dense_k=10,
    sparse_k=10,
):
    print(result.rrf_score, result.dense_rank, result.sparse_rank, result.chunk.text)

# Day 7 baseline: preserve the fused order while measuring both pipeline stages.
run = RerankingPipeline(index, IdentityReranker()).run(
    "How do I change MAX_POOL_SIZE?", candidate_k=20, top_k=5
)
for result in run.results:
    print(result.rerank_rank, result.rerank_score, result.chunk.source_name)
print(run.retrieval_latency_ms, run.reranking_latency_ms)

for candidate in index.detect_duplicates(threshold=0.95):
    print(candidate.first_chunk_id, candidate.second_chunk_id, candidate.similarity)
```

`result.chunk` contains the source-linked text and metadata. Dense scores are cosine similarities, sparse scores are BM25 ranking scores, and `rrf_score` is the fused rank score. None is an answer-confidence score or a benchmark result. `LocalIndexManager` writes `data/indexes/documind/chunk_manifest.jsonl` as the canonical chunk set and synchronizes BM25 and Chroma when chunks are added, deleted, or rebuilt. Near-duplicate review records live alongside the manifest and never remove source records automatically. Use `mode="dense"` or `mode="sparse"` on the same question to compare rankings.

### Reproducible TechQA retrieval comparison

After downloading and normalizing TechQA as described above, build the full local index and run the held-out comparison:

```powershell
python scripts/index_techqa.py
python scripts/evaluate_retrieval.py
```

The index uses fixed 1,000-character chunks with 150-character overlap and the pinned embedding revision from the project guide. The evaluator compares dense, sparse, and hybrid retrieval on the same held-out questions, with development-only warm-up. It reports document-level Recall@5/10, MRR@10, and mean/median/p95 retrieval latency. Results and full configuration are saved under `data/eval/results/` and are intentionally ignored by Git until they are reviewed. The report excludes unanswerable questions from recall and MRR and explains the benchmark's same-corpus evidence limitation. Do not claim a best method until the generated results have been inspected.

## Roadmap

| Milestone | Deliverable | Status |
| --- | --- | --- |
| Day 1 | Python package, normalized document model, Markdown/text/HTML/PDF loaders | Complete |
| Day 2 | Fixed, recursive, and semantic chunking with provenance | Complete |
| Day 3 | Pinned TechQA download, validation, normalization, and benchmark split | Complete |
| Day 4 | Pinned local embeddings and persistent ChromaDB add/query/delete/rebuild | Complete; CPU smoke test passes |
| Day 5 | BM25 search, shared chunk manifest, and reversible near-duplicate review | Complete |
| Day 6 | Dense, sparse, and hybrid retrieval with Reciprocal Rank Fusion | Complete; held-out comparison script added |
| Day 7 | Reranking and measurable latency | Implemented; 8 focused reranking tests pass |
| Days 8–9 | Grounded answer generation, citation checks, and safe abstention | Planned |
| Days 10–11 | Reviewed evaluation set, metrics, and retrieval/chunking experiments | Planned |
| Days 12–13 | FastAPI service and Streamlit product dashboard | Planned |
| Day 14+ | SQLite persistence, Docker, demo, documentation, and quality review | Planned |

The goal is a local-first MVP that can retrieve technical evidence, compare search approaches, show citations, abstain when support is weak, and report measured evaluation results. TechQA is a focused seed corpus and benchmark; it does not represent every company’s current documentation or prove generalization to other organizations.

## Screenshots

The product UI has not been built yet. Screenshots will be added when the Knowledge Library and Assistant screens are available.

<!-- Future: add a product screenshot or demo GIF here. -->

## Privacy and current limitations

- The current project is a local prototype for public or non-confidential material. Do not index confidential company documents.
- Authentication, access controls, tenant isolation, audit logs, retention controls, and prompt-injection defenses are not implemented.
- Scanned PDFs may not contain extractable text; OCR is not implemented.
- Held-out retrieval metrics are generated by `scripts/evaluate_retrieval.py`; inspect the local result artifact before making quality comparisons. Answer correctness and citation faithfulness are not yet benchmarked. No accuracy or production-readiness claims are made.
- Dataset-card licensing does not replace review of source-specific redistribution terms. Large raw datasets, processed corpora, indexes, model caches, and query logs should stay out of Git.

## Contributing

DocuMind AI is being developed in small, testable milestones. Please open an issue to discuss substantial changes, keep pull requests focused, add tests for new behavior, and document the checks you ran. Do not commit credentials, confidential documents, raw datasets, or generated index files.

## License

The project code license has not been selected yet. Until a license is added, permission to reuse or redistribute this repository’s code should not be assumed. Third-party datasets, models, and dependencies have their own licenses.

## Project guide

[PROJECT_GUIDE.md](PROJECT_GUIDE.md) is the living source of truth for the product scope, data handling, architecture, technical decisions, and day-by-day acceptance criteria.
