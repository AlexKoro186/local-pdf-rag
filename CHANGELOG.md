# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-19

### Added

- Local PDF extraction, normalization, deterministic overlapping chunks, and batch
  embeddings.
- Persistent cosine-similarity retrieval with Chroma and embedding-model compatibility
  checks.
- SHA-256-based incremental indexing that skips unchanged files and replaces changed
  sources.
- Streaming Streamlit chat with PDF uploads, source excerpts, page citations, relevance
  controls, runtime health, and conversation history.
- CLI commands for ingestion, querying, status inspection, and retrieval evaluation.
- Hit@K and MRR@K evaluation harness with a JSONL dataset format.
- Unit, integration, UI, and end-to-end checks with a 90% coverage gate.
- Docker Compose stack, GitHub Actions CI, environment configuration, security guidance,
  and architecture documentation.

### Security

- Document excerpts are delimited and treated as untrusted prompt data.
- Local PDFs, vector indexes, model data, secrets, and virtual environments are excluded
  from version control.
- The application container runs as a non-root user.
- Ollama traffic bypasses ambient system proxies.
