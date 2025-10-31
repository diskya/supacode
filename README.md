# CodexBot MVP (Offline-Friendly)

Minimal tooling to index a codebase, semantically retrieve relevant code chunks, and assemble LLM-ready context. Designed to work without internet and to be containerized later.

## Features (MVP)
- Indexes Python repositories into function/class chunks (AST-based; Tree-sitter optional later)
- Generates embeddings via Sentence-Transformers if available; falls back to a deterministic hashing embedder
- Stores metadata in SQLite and vectors in a NumPy file for offline use
- Hybrid retrieval: semantic search + simple symbol match fusion
- CLI: `index`, `query`, `chat` (optional HTTP call to your intranet LLM)

## Quickstart

- Python 3.10+
- Recommended optional deps: `numpy`, `tqdm`, `requests`, `sentence-transformers` (if you want neural embeddings)

```
# Create venv and install minimal deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # optional; or install numpy+tqdm+requests only

# Index a repo
python -m codexbot.cli index --repo /path/to/repo --index-dir .codexbot/index

# Search
python -m codexbot.cli query --index-dir .codexbot/index --q "how to parse config?" --top-k 8

# Chat (edit LLM endpoint via env vars or flags)
python -m codexbot.cli chat --index-dir .codexbot/index --q "Explain how the CLI builds the prompt"
```

## Environment
- `LLM_API_URL` (e.g. http://intranet-llm/api)
- `LLM_API_KEY` (optional)
- `LLM_MODEL` (optional)

## Notes
- MVP targets Python. You can add more languages by plugging in a Tree-sitter-based chunker later.
- The hashing embedder works fully offline with no model files; swap to a local `SentenceTransformer` path for better quality.

