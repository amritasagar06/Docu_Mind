# DocuMind AI — Product and Implementation Guide

**Status:** Living project source of truth
**Product stage:** Portfolio MVP / local-first prototype
**Primary dataset:** NVIDIA TechQA-RAG-Eval, hosted on Hugging Face
**Plan:** 14 implementation days, delivered as small reviewable slices

This guide governs product scope, technical choices, dataset handling, and acceptance checks. Work one day at a time. Before starting a day, inspect the existing repository and preserve its interfaces. Finish that day's acceptance checks before moving on. If implementation uncovers a constraint, update this guide and the README so the repository remains internally consistent.

## 1. Product vision

### Startup concept

**DocuMind AI helps technical teams find reliable answers in their documentation, with the evidence attached.**

Engineering and IT knowledge is distributed across support notes, runbooks, manuals, Markdown files, PDFs, and web pages. Keyword search can miss paraphrased questions, while vector search can overlook exact identifiers such as `MAX_POOL_SIZE`, `HTTP 422`, or `MQRC_CLUSTER_RESOLUTION_ERROR`. A general-purpose language model may sound certain even when the documents do not support its answer.

DocuMind AI is a documentation knowledge product built around inspectable evidence. It ingests documents, retrieves passages using both semantic and exact-term search, and generates answers tied to citations. When the evidence is weak or missing, it should say so. Analytics should show how retrieval and answer quality were measured.

### Intended users

- **Primary:** software engineers, IT support, and internal technical teams.
- **Secondary:** engineering leaders evaluating support knowledge workflows.
- **Portfolio audience:** technical reviewers evaluating a reproducible AI engineering project.

The MVP is a local prototype for public or non-confidential material. It is not a production multi-tenant company knowledge service.

### Product areas

1. **Knowledge Library:** add, inspect, index, and manage documents.
2. **AI Assistant:** ask questions, compare retrieval modes, inspect citations, and understand abstentions.
3. **Analytics & Evaluation:** inspect operational usage separately from offline benchmark results.

## 2. Dataset decision: TechQA-RAG-Eval

Use Hugging Face dataset **[`nvidia/TechQA-RAG-Eval`](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)** as the first real-world technical-support benchmark and seed corpus. Do not substitute StackSample unless this decision is revisited explicitly.

The dataset card describes real questions from IBM developer forums and associated technical documentation, and provides fields for an ID, question, answer, answerability flag, and context documents. The original TechQA research describes questions with accepted answers in IBM Technotes. This is a closer fit to technical-document retrieval than a generic Q&A dump.

### Known dataset facts and checks

- Hugging Face currently exposes a single `train` split. Treat it as a source collection, not as a ready-made train/dev/test protocol.
- The dataset page's viewer reports **910 rows**, while the card's quantification text says **908 question/answer pairs**. The card also reports **46 MB**, while the page shows roughly **49.8 MB** total. Do not hard-code either count. The dataset acquisition step must record the downloaded revision, observed files, row count, file sizes, and SHA-256 checksums.
- The page lists five fields: `id`, `question`, `answer`, `is_impossible`, and `contexts`. Each context may contain a filename and document text. Validate the exact downloaded schema before transformation.
- The dataset card labels the dataset **Apache-2.0**. Preserve the dataset URL, card revision, retrieval date, attribution, and license notice in `DATA_SOURCES.md`. Before publishing the raw corpus or derived extracts, inspect the upstream repository files and applicable terms; a dataset-card license label does not replace source-specific review.
- Keep the raw download immutable and outside ordinary Git history. Commit the acquisition and normalization scripts, a small synthetic test fixture, and metadata/checksums. Do not commit the raw archive by default.

### Dataset roles and leakage controls

Use the source in two related but distinct ways:

1. **Seed document corpus:** normalize the supplied context documents into canonical document records and index their text. Deduplicate by stable source filename/content hash while preserving which QA records reference each document.
2. **Offline RAG benchmark:** retain each question, reference answer, `is_impossible` value, and expected context IDs as evaluation-only records. Do not expose reference answers or benchmark labels to the chat UI or generation prompt.

All rows belong to one source split, so create a deterministic local development/evaluation partition with a fixed seed and document the method. Keep the held-out question IDs out of prompt tuning and manual examples. Because the benchmark's expected context documents are part of the source corpus, retrieval scores measure recovery of known evidence within that corpus; they do not by themselves prove generalization to unseen companies or documents. Report this limitation.

### Data layout and versioning

```text
data/
  raw/techqa/                 # downloaded upstream files; ignored by Git
  manifests/techqa.json       # tracked: immutable revision, retrieval date, files, sizes, hashes, row count
  processed/                  # generated canonical documents/chunks; ignored by Git
  indexes/                    # generated Chroma/BM25 data; ignored by Git
  eval/golden_questions.jsonl # reviewed small evaluation set; track only after license/provenance review
```

Pin the Hugging Face dataset to a resolved commit/revision rather than floating `main`. A refresh is a new manifest and a deliberate re-run of evaluation, not an in-place replacement of raw files.

### Canonical normalized records

- **Document:** stable document ID; source dataset and source record IDs; original filename; source URI; title/heading if present; extracted body; content type; metadata; content hash; license/source attribution.
- **Chunk:** stable chunk ID; document ID; text; ordinal; heading; page number if available; chunking strategy/version; token/character count; content hash.
- **BenchmarkCase:** question ID; question; category; reference answer; answerable/unanswerable label; expected document IDs; split; source revision.

Never infer a page number or heading that the source does not provide. Preserve source IDs and filenames so a result can be traced to the original record.

## 3. MVP capabilities

### Required for the 14-day MVP

- Load `.md`, `.txt`, `.html`, and `.pdf` files; preserve source metadata and PDF page numbers.
- Normalize TechQA-RAG-Eval's JSON and context documents; validate schema and provenance.
- Offer fixed-size-with-overlap, structure-aware recursive, and semantic chunking strategies.
- Generate local embeddings and persist them in ChromaDB.
- Build BM25 keyword search over the same canonical chunks.
- Keep dense and sparse indexes synchronized on ingest, delete, and rebuild.
- Flag likely duplicate chunks using configurable cosine similarity (initial threshold 0.95); preserve originals and make decisions reversible.
- Retrieve dense top 10 and sparse top 10, fuse with configurable Reciprocal Rank Fusion, rerank up to 20 candidates, and return up to 5 evidence chunks.
- Generate an answer grounded only in retrieved passages with numbered `[1]`, `[2]` citations.
- Verify citation-to-claim support with a transparent, inspectable method; abstain when evidence is weak or absent.
- Run a reviewed benchmark of at least 50 questions and report retrieval, correctness, faithfulness, citation, abstention, and latency measures separately.
- Compare chunking strategies and hybrid versus dense-only retrieval on the same fixed cases.
- Provide FastAPI endpoints: `POST /v1/ask`, `GET /v1/documents`, and `POST /v1/ingest`.
- Provide a Streamlit dashboard with Knowledge Library, Assistant, and Analytics/Evaluation sections.
- Persist document metadata, query logs, and evaluation runs in SQLite; provide Docker Compose and repeatable seeding.
- Document setup, data provenance, limitations, a short demo, and a case study using measured results only.

### Explicitly deferred until the MVP works

Qdrant or other vector database adapters, HyDE/query rewriting, incremental indexing, SSE streaming, Redis, authentication, multi-tenancy, hosted-model dependence, and fine-tuning. No feature is considered complete until its acceptance checks pass and its real status is reflected in the README.

## 4. Technical decisions

| Layer | MVP choice | Reason / boundary |
| --- | --- | --- |
| Runtime | Python 3.11+ | Typed, accessible ecosystem |
| Package/config | `pyproject.toml`, environment-based settings | Reproducible local setup; no secrets committed |
| Text and HTML | Python standard library, BeautifulSoup | Small local loader dependencies |
| PDF | PyMuPDF | Text extraction with page-level provenance; scanned-PDF OCR is deferred |
| Dataset access | Hugging Face Hub / Datasets utilities | Source lives on Hugging Face; pin revision and record files before ingest |
| Embeddings | Sentence Transformers, initially `all-MiniLM-L6-v2` | Local inference; review model license and hardware before release |
| Dense index | ChromaDB persistent local store | MVP vector search and local persistence |
| Sparse index | `rank-bm25` | Exact terms and identifiers |
| Fusion | Small in-house RRF implementation | Explicit, testable rank-list fusion |
| Reranker | Start with a deterministic baseline; optional local cross-encoder after baseline | Avoid hiding retrieval behavior; measure latency and quality before enabling |
| Answer model | Ollama with a locally runnable instruction model | Free-first local generation; model/license choice must be checked at implementation time |
| API | FastAPI + Uvicorn | Typed REST interface |
| UI | Streamlit | Fast MVP for library, chat, and analytics |
| Metadata/logs | SQLite | Simple local transactional persistence |
| Tests | pytest | Unit and integration coverage |
| Packaging | Docker + Docker Compose | Reproducible demo and persisted data volumes |

Use official project documentation when implementing each integration. Pin compatible dependency ranges in project files, keep model names and thresholds configurable, and do not make the application dependent on a paid API.

## 5. Architecture and data flow

```text
Hugging Face TechQA-RAG-Eval             User-provided public documents
 JSON rows + context files                MD / TXT / HTML / PDF
              │                                    │
              └───────────────┬────────────────────┘
                              ▼
       source manifest → validation → normalization → provenance
                              ▼
       chunking (fixed / recursive / semantic) → duplicate review
                              │
                 ┌────────────┴─────────────┐
                 ▼                          ▼
        local embeddings                 BM25 corpus
        persistent ChromaDB              canonical chunks
                 └────────────┬─────────────┘
                              ▼
 Question → dense + sparse retrieval → RRF → reranking → top evidence
                              ▼
         local LLM answer with numbered citations or abstention
                              ▼
  claim/citation checks → API → Streamlit; logs/metadata in SQLite

Offline benchmark → fixed questions + expected documents/answers
                  → retrieval, answer, citation, abstention, latency reports
```

The offline benchmark is kept separate from interactive query logs. A benchmark label is never treated as evidence supplied to answer generation.

## 6. Fourteen-day implementation plan

Each day should produce working code, tests, an updated explanation, and a short checkpoint. Do not create empty future modules. “Done” means the acceptance checks pass on the target environment or remaining environment-specific failures are stated honestly.

### Day 1 — Project setup and document loaders — **complete**

**Build:** Python package, canonical `Document` model, Markdown/TXT/HTML/PDF loaders, project README, dependency setup, and loader unit tests.

**Accept when:** each supported format loads; headings and source metadata are preserved where available; PDF page numbering is one-based; empty files have defined behavior; missing, unsupported, encrypted, and malformed inputs raise clear errors; tests pass.

### Day 2 — Chunking strategies — **complete**

**Build:** shared `Chunk` model and configuration; fixed-size overlap chunker; structure-aware recursive chunker; semantic chunker using an injected similarity interface; preserve document, heading, and page provenance. The semantic similarity interface is intentionally ready for the local embedding work on Day 4; Day 2 itself does not load a model.

**Implemented:** `FixedSizeChunker` uses a configurable character limit and overlap, preferring whitespace boundaries. `RecursiveChunker` recognizes Markdown heading paths and recursively uses paragraph, line, sentence, and word boundaries before splitting an unbroken token. `SemanticChunker` groups adjacent paragraphs when an injected similarity function meets the configurable threshold and enforces the maximum character limit. All chunkers preserve source metadata and produce stable IDs.

**Verified:** `python -m pytest` passes 16 tests total, including the eight loader tests and eight chunking tests.

**Accept when:** boundaries and overlap are tested; short/empty/long inputs behave predictably; metadata survives chunking; semantic thresholds and chunk sizes are configurable; tests cover all strategies. **Status: met.**

### Day 3 — TechQA acquisition and normalization — **complete**

**Build:** repeatable downloader for the pinned Hugging Face revision; the `data/` layout above; schema validator; normalizer for QA rows and context documents; content hashing and deterministic dedupe; `DATA_SOURCES.md`; separate corpus and benchmark records; deterministic development/evaluation partition.

**Implemented:** `scripts/download_techqa.py` downloads or verifies `train.json` and `corpus.zip` at immutable revision `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`. `scripts/normalize_techqa.py` verifies local source checksums, validates the exact required schema, reads corpus files without extracting the archive, merges exact duplicate content by SHA-256, resolves every benchmark context filename to a stable document ID, and writes separate document and benchmark JSONL files. Benchmark cases receive a stable 80/20 development/evaluation partition using seed 42. Raw and processed data are ignored by Git; the source and normalized output checksums are recorded in `data/manifests/techqa.json`.

**Observed snapshot:** 4,019,332-byte `train.json` with SHA-256 `69d97231509482ed6bd5ec1c4bc0607acb82a88d11169eb8383592d0ca8b93c7`; 45,814,227-byte `corpus.zip` with SHA-256 `c06aa287dcc1abf8db6b49b8495df095db73342d729f6451ac330785245d10be`; 910 questions; 300 unanswerable questions; 28,481 corpus files; 28,478 unique document contents after collapsing 3 duplicates; 496 distinct referenced filenames; 0 unresolved context filenames; 728 development and 182 evaluation cases. The three files that contain a literal Unicode replacement character are valid UTF-8, so no decoding replacement was needed.

**Verified:** eight dataset tests pass. They cover schema failures, duplicate IDs/content, source-to-benchmark separation, repeatable output, deterministic splits, missing linked documents, invalid UTF-8 warning behavior, and unsafe archive paths. The normalizer ran successfully on the pinned dataset.

**Accept when:** a clean run records revision, observed row count, file sizes, checksums, and license source; repeated runs produce identical normalized IDs; malformed rows are reported, not silently skipped; benchmark answers/labels cannot enter indexed text or answer prompts; sample fixtures work without downloading the full dataset. **Status: met.**

### Day 4 — Local embeddings and ChromaDB

**Build:** embedding interface and local Sentence Transformers implementation; persistent Chroma collection; stable chunk IDs and metadata; add/query/delete/rebuild operations.

**Accept when:** embeddings are reproducible for the configured model; vector results map back to canonical chunks and source IDs; persistence survives process restart; delete/rebuild leave the index consistent; CPU-only run is documented. **Status: met.**

**Implemented:** `documind.embeddings.EmbeddingModel` and the lazy CPU-first `SentenceTransformerEmbedder` use the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` model at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; normalized embeddings are requested from the model. `documind.vectorstore.ChromaVectorStore` stores vectors and reconstructable chunk provenance in a persistent cosine-distance Chroma collection. Stable chunk IDs are upsert keys; queries return canonical chunks plus distance and similarity; delete reports removed IDs; rebuild first stages a replacement collection and then swaps it into the active name. Collection metadata records the model/revision/dimension and refuses non-empty collections created with a different model revision. Tests use a tiny deterministic embedder; model downloads are not needed for tests. `scripts/smoke_test_vectorstore.py` runs the pinned real model on CPU, performs a semantic query, then launches a fresh process to confirm the index is persisted. **Verified:** 33 tests pass; the CPU smoke test passed with the real model and ChromaDB 1.5.9. The first smoke test run needs internet to fetch the model weights.

### Day 5 — BM25 and duplicate handling

**Build:** BM25 index on canonical chunks; explicit rebuild/persistence strategy; configurable near-duplicate detection; reversible duplicate status.

**Accept when:** exact identifiers and error codes are searchable; BM25 and vector indexes derive from the same chunk manifest; add/delete/rebuild updates both; duplicate threshold is configurable and does not destroy source records.

**Implemented:** `ChunkManifestStore` atomically persists canonical chunks to JSONL. `LocalIndexManager` treats this manifest as the source of truth and keeps Chroma and `BM25SearchIndex` synchronized on add, delete, rebuild, and reopen; it repairs a stale Chroma ID set from the manifest rather than silently trusting it. BM25 uses `rank-bm25` `BM25Okapi` with a case-insensitive tokenizer that retains identifiers such as `MAX_POOL_SIZE` and `MQRC_CLUSTER_RESOLUTION_ERROR` as single terms. `DuplicateDetector` searches indexed nearest neighbors in a batch, defaults to cosine similarity 0.95 and 10 candidates per chunk, and saves pair IDs/scores in `DuplicateRegistry`. Reviews can be set to `review`, `duplicate`, or `distinct`; status changes are reversible, and the original chunks remain in both indexes. **Verified:** `python -m pytest -q` passes 38 tests, including exact identifier/error-code search, persistent manifest reload, synchronization on add/delete/rebuild, stale-vector repair, duplicate threshold detection, and reversible review decisions. `python -m compileall -q src\\documind scripts tests` and `git diff --check` completed without errors. **Status: met.**

### Day 6 — Dense, sparse, and hybrid retrieval

**Build:** dense and sparse retrievers; configurable top-k; Reciprocal Rank Fusion with configurable constant; result schema with ranks and scores.

**Accept when:** each retriever returns source-linked chunks; RRF handles overlap, ties, and missing results deterministically; dense-only and hybrid modes can be compared using the same question set.

**Implemented:** `ReciprocalRankFusion` combines dense `SearchResult` and sparse `BM25Result` lists using `sum(1 / (constant + one_based_rank))`; the constant defaults to 60 and is configurable through `LocalIndexManager` or `HybridRetriever`. Ties sort by best contributing rank and then stable chunk ID. Repeated IDs in a channel count only at their first rank. `RetrievalResult` includes the canonical chunk, mode, fused score, per-channel ranks, original dense/sparse scores, and RRF score. `LocalIndexManager.search()` exposes `dense`, `sparse`, and `hybrid` modes with configurable `dense_k`, `sparse_k`, and `top_k`, so the same question can be compared across modes. **Verified:** `python -m pytest -q` passes 44 tests; focused coverage includes overlap, missing channels, ties, duplicate IDs, score provenance, configurable RRF, and source-linked results from each mode. Compilation and `git diff --check` passed. **Status: met.**

### Day 7 — Reranking — **implemented; benchmark run in progress**

**Build:** baseline reranker interface; optional local cross-encoder; retrieve up to 10 dense + 10 sparse, fuse, rerank up to 20, retain up to 5; latency and score provenance.

**Implemented:** `IdentityReranker` provides a no-model baseline that preserves retrieval order and scores. `CrossEncoderReranker` is lazy, requires a pinned model revision before loading, and can take an injected model for tests. Its default model name is `cross-encoder/ms-marco-MiniLM-L6-v2` (the model card identifies Apache-2.0 licensing and query-passage reranking use); model weights are not fetched by import or tests. `RerankingPipeline` requests 10 dense + 10 sparse candidates, fuses at most 20, returns up to 5 by default, and reports retrieval and reranking latency separately. Each result retains its original `RetrievalResult`, including chunk/document/source IDs, retrieval channel ranks, channel scores, and RRF score, alongside rerank rank and score. The optional model is not enabled by default; pin and record a specific model revision before using it for reproducible comparisons.

**Corpus evaluation preparation:** `scripts/index_techqa.py` applies fixed chunks of 1,000 characters with 150 overlap to the pinned 28,478-document corpus, assigns canonical TechQA document IDs, and records the dataset/document hashes and model revision in an ignored local index config. Chroma upserts embeddings in bounded batches. `scripts/evaluate_retrieval.py` compares dense, BM25 sparse, and RRF hybrid on the same held-out local evaluation cases. It warms the embedding model only with development questions, supplies only question text to retrieval, rotates mode order, and computes document-level Recall@5/10, MRR@10, and mean/median/p95 latency. Per-case and summary JSON are written under ignored `data/eval/results/`; the summary records denominators, configuration, and limitations. This is retrieval-only evaluation; it does not report answer correctness or abstention quality.

**Accept when:** reranking can be disabled; input/output IDs remain traceable; top-k and latency are measured; optional model is not downloaded unexpectedly during tests. **Code acceptance: met; held-out retrieval comparison follows after full-corpus indexing completes.**

### Day 8 — Grounded answer generation

**Build:** Ollama client interface, versioned prompt, numbered evidence blocks, structured answer response, explicit no-evidence behavior.

**Accept when:** the model receives only the question and retrieved evidence; citations refer only to provided blocks; insufficient context produces an abstention; malformed model output is handled; prompts are tested with a stub client.

### Day 9 — Citation verification and confidence

**Build:** claim/citation association; deterministic checks for citation IDs and source existence; support verification method; confidence components; feedback record format.

**Accept when:** invalid citation references are rejected; unsupported claims can be flagged; confidence is labeled heuristic unless calibrated; tests include supported, unsupported, and mixed claims; verifier limitations are documented.

### Day 10 — Evaluation dataset and metrics

**Build:** benchmark loader for TechQA-derived cases; at least 50 hand-checked cases across factual, paraphrase, exact identifier, multi-hop where available, unanswerable, ambiguous, and troubleshooting types; Recall@k, correctness, faithfulness, citation accuracy, abstention quality, and latency.

**Accept when:** benchmark cases have expected facts and source IDs; metrics have written definitions; deterministic split and source revision are recorded; reports disclose sample size and known dataset limitations; no fabricated ground truth or results.

### Day 11 — Chunking and retrieval experiments

**Build:** experiment runner applying identical benchmark cases to all three chunkers and hybrid vs dense-only retrieval; machine-readable result file and concise report.

**Accept when:** runs are reproducible from documented settings; comparison includes retrieval metrics and latency; selected defaults are based on actual results; empty/failed cases are visible.

### Day 12 — FastAPI service

**Build:** `POST /v1/ask`, `GET /v1/documents`, and `POST /v1/ingest`; typed request/response schemas; upload limits and clear validation errors; health endpoint if useful.

**Accept when:** API tests cover valid, invalid, unanswerable, and empty-index requests; response citations resolve to evidence; ingestion failure status is visible; no secrets or local paths leak unnecessarily.

### Day 13 — Streamlit product dashboard

**Build:** Knowledge Library, AI Assistant, and Analytics/Evaluation screens; upload supported files; hybrid/dense-only control; citation expanders; evidence and abstention display; operational analytics separate from offline benchmark.

**Accept when:** user can ingest, inspect, query, follow citations, understand abstention, and view only actually collected metrics; UI errors are understandable; core flow works from a clean local start.

### Day 14 — Persistence, Docker, quality, demo, and case study

**Build:** SQLite records and query logs; repeatable seed script; Dockerfile/Compose volumes; `.env.example`; final audit; demo under four minutes; case study and limitations.

**Accept when:** clean documented startup works; Chroma and SQLite data survive restart; seed is repeatable; no credentials or large raw corpus are committed; tests and benchmark run or known gaps are documented; demo shows ingest, exact and semantic questions, citations, abstention, and retrieval comparison; all case-study metrics come from actual runs.

## 7. Quality and evaluation rules

- **Recall@k:** share of expected evidence document IDs present in the top k results.
- **Answer correctness:** expected factual elements present in the answer, using a documented human/judge scoring method.
- **Faithfulness:** share of answer claims supported by retrieved evidence.
- **Citation accuracy:** share of citations that support the claims they accompany.
- **Abstention quality:** correct abstention on unanswerable/underspecified cases and answers when support exists.
- **Latency:** report retrieval, reranking, generation, verification, and total latency separately where available.
- Keep operational usage analytics separate from fixed-set offline evaluation.
- An LLM verifier is an aid, not a guarantee. Evaluate it against human-checked examples.
- Do not call heuristic confidence a probability unless calibration has been measured.
- Never claim accuracy, production readiness, or a benchmark win without actual reproducible evidence.

## 8. Security, licensing, and operational limits

- Do not upload confidential company documents to the MVP.
- Before real deployment, add authentication, authorization, tenant isolation, audit logs, retention controls, prompt-injection defenses, and secure secret management.
- Treat document text as untrusted input. Retrieved passages must not be allowed to change system policy or reveal secrets.
- Scanned PDFs may yield no text; surface an extraction warning. OCR is outside the first MVP.
- Deduplication flags must be reviewable and reversible because repeated technical content can be useful.
- Keep `.env`, raw data, processed corpora, indexes, model caches, and query logs out of Git unless a reviewed small fixture is intentionally included.
- Review licenses for the dataset, source documents, dependencies, and models separately. Project code license remains a separate decision.

## 9. Definition of done

A reviewer can clone the repository, follow the README, acquire or seed the documented sample, inspect indexed documents and chunks, ask semantic and exact-identifier questions, compare hybrid with dense-only retrieval, inspect supporting passages and citation sources, receive a safe abstention, run the benchmark and chunking comparison, and inspect analytics derived from persisted records. The case study accurately states what is implemented and what is still limited.

## 10. Working protocol

For every daily task:

1. Inspect current files and this guide; state the proposed file/interface changes before editing.
2. Implement only the day's scope, reusing existing interfaces.
3. Add unit tests for new behavior and integration tests for important flows.
4. Run the focused tests, then relevant broader checks; report exact commands and results.
5. Update README, this guide, and data-source notes when scope, behavior, or decisions change.
6. Summarize changed files, verification, remaining risks, and the next day's task.

Do not start the next day's scope while the current day's acceptance criteria remain unresolved unless the user deliberately reprioritizes the plan.

## Sources

- [NVIDIA TechQA-RAG-Eval on Hugging Face](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval) — dataset card, schema, files, license label, and current viewer inventory.
- [TechQA paper, ACL Anthology](https://aclanthology.org/2020.acl-main.117/) — original dataset description and provenance.
- [NVIDIA technical blog on retriever evaluation](https://developer.nvidia.com/blog/evaluating-retriever-for-enterprise-grade-rag/) — technical-support retrieval context.

Re-check the dataset card and source terms when acquiring data; the repository's revision and observed manifest should be recorded with each reproducible run.
