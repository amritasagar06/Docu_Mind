Data sources

## NVIDIA TechQA-RAG-Eval

- **Dataset:** [`nvidia/TechQA-RAG-Eval`](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)
- **Use:** Public technical-support seed corpus and offline RAG benchmark.
- **Provenance:** Dataset card describes real IBM developer-forum questions with answers linked to technical documentation; original TechQA research describes IBM Technote sources.
- **License label:** Apache-2.0 on the Hugging Face dataset card. Preserve attribution and verify source-document terms before redistribution.
- **Expected fields:** `id`, `question`, `answer`, `is_impossible`, `contexts` (validate the actual downloaded schema).
- **Acquisition status:** Downloaded and normalized for local development. Raw files and processed JSONL are excluded from Git.
- **Pinned revision:** `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`
- **Retrieved:** 2026-11-26 UTC.
- **Observed inventory:** 910 question records; 28,481 text files in `corpus.zip` (104,882,784 uncompressed bytes); 300 questions marked unanswerable; 496 distinct context filenames, all matched in the corpus. Three exact duplicate document contents were collapsed, leaving 28,478 normalized documents.
- **Local question split:** deterministic SHA-256 ordering with seed 42; 728 development cases and 182 evaluation cases. This is a local partition of the source `train` split, not an official benchmark split.
- **Checksums and exact file sizes:** recorded in [`data/manifests/techqa.json`](data/manifests/techqa.json), alongside the pinned file URLs, normalization statistics, and checksums of normalized outputs.
- **Decode note:** corpus documents passed strict UTF-8 decoding; no invalid byte sequences required replacement.

Do not commit the raw corpus by default. Keep benchmark answers and labels out of interactive answer generation. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for normalization, evaluation, and leakage rules.
