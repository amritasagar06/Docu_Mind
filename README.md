# Docu_Mind — DocuMind AI

> **Find answers in technical documentation, with the evidence attached.**

DocuMind AI is a documentation knowledge assistant in active development. The goal is to help engineering and IT teams find reliable answers across technical documents and inspect the passages behind each answer. The first milestone builds the ingestion foundation; search, answer generation, and the product interface are on the roadmap. The complete product, dataset, architecture, and day-by-day implementation requirements live in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Why DocuMind AI

Technical knowledge is spread across Markdown guides, PDFs, web pages, and Q&A archives. Keyword search can miss paraphrases, while semantic search can overlook exact identifiers such as `MAX_POOL_SIZE`, `HTTP 422`, or `Depends()`. A language model can also produce a confident answer that the source material does not support.

DocuMind AI is being built around a practical principle: answers should be grounded in retrieved source passages, and users should be able to inspect that evidence. The project combines document ingestion, hybrid retrieval, citations, and evaluation rather than treating a chat window as the whole product.

## Product areas

- **Knowledge Library:** ingest, inspect, and manage technical documents.
- **AI Assistant:** ask questions, inspect cited passages, and see when evidence is insufficient.
- **Analytics & Evaluation:** measure retrieval quality, answer faithfulness, citation quality, abstentions, and latency.

These are the intended product areas. The Day 1 ingestion foundation, Day 2 chunking layer, and Day 3 TechQA acquisition/normalization pipeline are implemented; retrieval and the product interface remain on the roadmap.

## Features

### Available in the Day 1 foundation

- Load `.md`, `.txt`, `.html`, `.htm`, and `.pdf` files.
- Normalize results into a typed `Document` record with source URI, source name, content type, headings, and optional page number.
- Keep PDF pages as separate records with one-based page numbers for future citations.
- Raise a clear `DocumentLoadError` for unsupported, missing, unreadable, encrypted, or malformed sources.
- Split documents with fixed-size overlap, Markdown-aware recursive, or similarity-driven semantic chunking.
- Preserve stable chunk IDs, source/document identity, source headings, page numbers, content type, and custom metadata.
- Download a pinned TechQA-RAG-Eval revision, verify source checksums, and normalize documents separately from evaluation questions and answers.
- Keep benchmark reference answers and answerability labels out of the normalized document corpus.

### Planned

- Local embeddings, ChromaDB vector search, and BM25 keyword search.
- Reciprocal Rank Fusion (RRF), reranking, grounded answers, and citation checks.
- FastAPI service, Streamlit dashboard, SQLite metadata, and reproducible evaluation.

## Technology

| Area | Planned technology | Status |
| --- | --- | --- |
| Language | Python 3.11+ | Project foundation |
| Markdown/text and HTML extraction | Python standard library, BeautifulSoup | Day 1 |
| PDF extraction | PyMuPDF | Day 1 |
| Chunking | Project-native fixed, recursive, and semantic strategies | Day 2 |
| Local embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | Planned |
| Vector retrieval | ChromaDB | Planned |
| Keyword retrieval | `rank-bm25` | Planned |
| Local language model | Ollama | Planned |
| API and dashboard | FastAPI, Streamlit | Planned |
| Metadata and logs | SQLite | Planned |
| Evaluation | pytest and a reviewed golden-question set | Planned |

The intended MVP is local-first and free-first. Local model requirements and model licenses will be reviewed before those components are selected for release.

## Seed dataset

The selected first real-world corpus and evaluation source is [`nvidia/TechQA-RAG-Eval`](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval), a Hugging Face dataset based on technical-support questions from IBM developer forums and their associated technical documents. The pinned local snapshot contains 910 questions and 28,478 unique documents after exact-content deduplication. The dataset card labels it Apache-2.0. See [DATA_SOURCES.md](DATA_SOURCES.md) and [`data/manifests/techqa.json`](data/manifests/techqa.json) for provenance and the observed checksums.

## How it works

```text
Current (Days 1–3)
Local file ──> format loader ──> normalized Document record(s)
                                  └─ PDF: one record per page
Document ──> fixed / recursive / semantic chunker ──> source-linked chunks
Pinned TechQA files ──> checksum/schema validation ──> documents + separate benchmark cases

Planned end-to-end workflow
Documents ──> extraction ──> chunking ──┬─> local embeddings ──> ChromaDB
                                         └─> BM25 keyword index
Question ──> dense + keyword retrieval ──> RRF / reranking
          ──> evidence-grounded answer ──> citations + verification
```

## Quick start

### Requirements

- Python 3.11 or newer
- `pip`

### Install

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

### Load a document

```python
from pathlib import Path

from documind.ingestion import load_document

records = load_document(Path("docs/guide.pdf"))
for record in records:
    print(record.source_name, record.page_number, record.text[:100])
```

For a source checkout before installing the package, set `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH = "src"`). A PDF yields one record per page. Other supported files yield one record per file. Scanned PDFs may have no extractable text; OCR is outside the current scope.

### Acquire and normalize TechQA

```powershell
python scripts\download_techqa.py
python scripts\normalize_techqa.py
```

The downloader uses the pinned revision and writes source checksums to `data/manifests/techqa.json`. Normalization writes `data/processed/techqa_documents.jsonl` and `data/processed/techqa_benchmark_cases.jsonl`. Raw and processed data remain out of Git.

### Create chunks

```python
from pathlib import Path

from documind.chunking import ChunkingConfig, FixedSizeChunker
from documind.ingestion import load_document

document = load_document(Path("docs/guide.md"))[0]
chunker = FixedSizeChunker(ChunkingConfig(max_chars=800, overlap_chars=120))
chunks = chunker.chunk(document)
print(chunks[0].chunk_id, chunks[0].heading, chunks[0].text[:100])
```

The semantic chunker accepts a similarity function so it can be tested independently of a model. Local embedding integration is planned for a later day; the current semantic strategy does not download or load an embedding model.

## Verify the Day 1 foundation

```powershell
python -m pytest
```

The tests use small temporary files and generate a tiny PDF locally. They do not download a corpus or call an external model.

## Roadmap

| Milestone | Focus | Status |
| --- | --- | --- |
| Day 1 | Python package, normalized document model, Markdown/text/HTML/PDF loaders | Complete |
| Day 2 | Shared chunk model and fixed, structure-aware, and semantic chunking | Complete |
| Day 3 | TechQA acquisition, normalization, and provenance | Complete |
| Days 4–7 | Local embeddings, ChromaDB, BM25, deduplication, RRF, and reranking | Next |
| Days 8–11 | Grounded answers, citation verification, evaluation, and experiments | Planned |
| Days 12–14 | API, dashboard, persistence, Docker, quality pass, and demo | Planned |

The MVP target is a reproducible technical-support corpus, inspectable citations, an explicit abstention path, and measured retrieval and answer quality. TechQA is a focused benchmark and seed corpus, not a substitute for a broad set of current company manuals. Its age, scope, and benchmark limitations will be disclosed in evaluation reports.

## Screenshots

**Product screenshots will be added after the Knowledge Library and Assistant interfaces are implemented.** The current milestones are a Python ingestion and chunking library; there is no graphical interface yet.

## Data, licensing, and contributions

No third-party corpus is bundled in this repository. Before adding datasets or public documentation, record the source, access date, and applicable license in `DATA_SOURCES.md`; review attribution and redistribution terms. Keep large raw datasets and private documents out of Git.

**Project license: not selected yet.** Until a license is added, no permission to reuse or redistribute this project's code should be inferred. Dependencies and any future datasets have their own licenses.

Contributions are welcome as the project takes shape. Please open an issue to discuss larger changes, keep pull requests focused, include tests for new behavior, and report only checks that were actually run. Never commit credentials or confidential documents.

## Project status and limitations

This is an early portfolio/MVP build, not a production-ready company knowledge service. Authentication, authorization, tenant isolation, audit logs, retention controls, and prompt-injection defenses are not implemented. Do not upload confidential company documents. Search and answer quality will be reported only after the corresponding evaluation work exists.
