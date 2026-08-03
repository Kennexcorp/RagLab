# RAGLab

**A hands-on lab for building intuition about Retrieval Augmented Generation — ingest real documents, then compare 10 chunking strategies against 8 retrieval strategies and watch how each choice changes what the model retrieves and answers.** Built with LangChain, ChromaDB, and HuggingFace embeddings; exposed via a Streamlit chat UI, CLI, and MCP server.

## What it is

Most RAG tutorials pick one chunking method and one retrieval strategy, wire them together, and call it done — so you never actually see *why* one choice beats another. Answering "would hybrid search have found this better than semantic search?" or "does markdown-aware chunking actually help here?" normally means writing a new throwaway script per combination.

RAGLab turns that into a live comparison instead of a guessing game: ingest a document or dataset once, then swap the chunking strategy and retrieval strategy from the sidebar (or a CLI flag) and re-ask the same question. Every answer comes back with its cited source chunks, so you can see exactly which passages a given strategy pulled in — and why the answer changed when you switched it.

## Use case

You drop a folder of PDFs, DOCX reports, or JSON/CSV exports into the **Upload Documents** tab. You ask: *"What drove the Q3 revenue dip?"*

- With `semantic` retrieval + `fixed` chunking, the answer cites two loosely related paragraphs split mid-sentence.
- Switch chunking to `markdown-headers` (chunks respect section boundaries) and retrieval to `hybrid` (semantic + BM25 keyword matching) — same question, and now the cited chunks are the actual "Q3 Financial Summary" section, with a more grounded answer.
- Try `hyde` (the LLM drafts a hypothetical answer first, then searches by that) or `multi-query` (the LLM rephrases your question several ways and merges results) to see how query-side techniques compare to chunking-side ones.

That side-by-side comparison — same data, same question, different pipeline — is the whole point.

## Key Features

- **Streamlit Chat Interface**: Persistent multi-turn conversation with per-collection history
- **Multi-Collection Support**: Organize documents into named collections; switch collections in the sidebar
- **8 Retrieval Strategies**: Semantic, Hybrid, MMR, Multi-Query, Reranking, HyDE, Self-Query, Parent-Child
- **10 Chunking Strategies**: Semantic, Fixed, Markdown, Markdown-Headers, Token, Python, LaTeX, SpaCy, NLTK, Semantic-Embedding
- **Multiple LLM Providers**: Ollama (default/local), OpenAI, Anthropic Claude
- **Document Ingestion**: PDF, DOCX, TXT, JSON, CSV via LangChain loaders
- **Source Citations**: Answers include references to retrieved source chunks — the mechanism that makes strategy comparison visible
- **MCP Integration**: Expose as MCP tools for integration with Claude Desktop and Claude Code

## Quick Start

### 1. Installation

```bash
cd RAGLab

# Install dependencies and sync virtual environment
uv sync
```

### 2. Configuration

```bash
cp .env.example .env
```

Edit `.env` — minimum required for local use with Ollama:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

For cloud LLMs:
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_key_here
LLM_MODEL=claude-sonnet-4-6
```

### 3. Run the GUI

```bash
uv run streamlit run gui_app.py
```

The app opens at `http://localhost:8501`. Use the **Upload Documents** tab to ingest files, then switch to **Chat** to ask questions — and try re-asking the same question after changing chunking/retrieval strategy in the sidebar.

### 4. CLI Usage

```bash
# Ingest a JSON or CSV file
uv run python -m raglab.rag_system --ingest data.json --source-type json

# Ask a one-shot question
uv run python -m raglab.rag_system --query "What does this document say about revenue?"

# Interactive REPL
uv run python -m raglab.rag_system --interactive

# View system stats
uv run python -m raglab.rag_system --stats

# Clear a collection
uv run python -m raglab.rag_system --clear
```

## Docker

Prebuilt images are published to GitHub Container Registry for `linux/amd64` and
`linux/arm64` (Apple Silicon pulls the arm64 variant automatically). The embedding model,
cross-encoder reranker, spaCy model, and NLTK data are all baked in, so the container
starts without downloading anything.

```bash
docker pull ghcr.io/kennexcorp/raglab:latest

# GUI on http://localhost:8501, with ingested data persisted to the host
docker run --rm -p 8501:8501 \
  -v "$PWD/vector_db:/app/vector_db" \
  -v "$PWD/data:/app/data" \
  --env-file .env \
  ghcr.io/kennexcorp/raglab:latest
```

The default command is the Streamlit GUI. Override it for the other entry points:

```bash
# CLI (interactive REPL needs -it)
docker run --rm -it -v "$PWD/vector_db:/app/vector_db" \
  ghcr.io/kennexcorp/raglab:latest python -m raglab.rag_system --interactive

# MCP server (stdio transport, no port)
docker run --rm -i -v "$PWD/vector_db:/app/vector_db" \
  ghcr.io/kennexcorp/raglab:latest python mcp_server.py
```

Mount `/app/vector_db` and `/app/data` or ingested collections are lost when the container
is removed.

**Using Ollama from the container:** `localhost` inside the container is not your host, so
set `OLLAMA_BASE_URL=http://host.docker.internal:11434` (Docker Desktop) or point it at an
Ollama service on the same Docker network. Cloud providers need no extra setup beyond the
API key in `.env`.

**Air-gapped hosts:** the default embedding model, reranker, spaCy model, and NLTK data
are cached in the image, but HuggingFace still makes a revision check on load and errors
out if the network is unreachable rather than absent-but-routable. Add `-e HF_HUB_OFFLINE=1`
to skip that check and load purely from cache. It is not the default because
`EMBEDDING_MODEL` and `RERANKER_MODEL` are configurable, and offline mode would stop you
pulling a model that is not already baked in.

### Image tags

Images are published on tagged releases only.

| Tag | Points at |
|---|---|
| `latest` | Most recent stable release |
| `1.2.3` | A specific release |
| `1.2` | Latest patch on that minor line |

Pre-release tags (`v1.2.3-rc1`) publish only the exact version, leaving `latest` and the
`{{major}}.{{minor}}` line pointing at the last stable release.

Building locally:

```bash
docker build -t raglab:dev .
```

## GUI Overview

### Sidebar Controls

| Section | Controls |
|---|---|
| **Collections** | Switch active collection; create, clear, or delete collections |
| **LLM Settings** | Provider (Ollama/OpenAI/Anthropic), model name, API key, temperature |
| **System Prompt** | Override the default assistant prompt |
| **Chunking Settings** | Strategy selector (10 options), chunk size, chunk overlap |
| **Retrieval Settings** | Strategy selector (8 options), Top-K, semantic weight (hybrid only) |
| **Stats** | Document count and embedding model for the active collection |

### Upload Documents Tab

Upload PDF, DOCX, TXT, JSON, or CSV files. Supports multi-file upload. Form resets after each ingestion.

### Chat Tab

- Persistent conversation per collection — follow-up questions carry context from prior turns
- Query rewriting: vague follow-ups ("explain that") are rewritten into standalone search queries before retrieval
- Each assistant response includes a collapsible **Sources** expander with similarity scores and text previews — this is the comparison surface: re-run a question after changing strategy and diff the sources
- **Clear chat** button resets the current conversation without deleting documents

## Retrieval Strategies

| Strategy | Description |
|---|---|
| `semantic` | Dense vector similarity search (default) |
| `hybrid` | Semantic + keyword (BM25-weighted ensemble); adjustable semantic weight |
| `mmr` | Maximal Marginal Relevance — diverse, non-redundant results |
| `multi-query` | LLM generates multiple query phrasings; results merged |
| `reranking` | Initial semantic recall + cross-encoder reranking |
| `hyde` | Hypothetical Document Embeddings — LLM drafts an answer, searches by that |
| `self-query` | LLM extracts metadata filters from the question |
| `parent-child` | Retrieves small child chunks; returns larger parent chunks for context |

## Chunking Strategies

| Strategy | Description |
|---|---|
| `semantic` | Splits on sentences/paragraphs by character count (default) |
| `fixed` | Fixed character count with overlap |
| `markdown` | Splits on markdown headers and sections |
| `markdown-headers` | Splits strictly on header hierarchy |
| `token` | Splits by token count (tiktoken) |
| `python` | AST-aware splitting for Python source files |
| `latex` | LaTeX-aware splitting (sections, equations) |
| `spacy` | Sentence boundary detection via spaCy NLP |
| `nltk` | Sentence tokenization via NLTK |
| `semantic-embedding` | Groups semantically similar sentences using embeddings |

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit GUI                      │
│   Upload Tab              Chat Tab                   │
│   (ingest docs)    (multi-turn conversation)         │
└───────────────────────────┬─────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   RAGSystem      │
                   │  orchestrator   │
                   └────┬───────┬───┘
                        │       │
           ┌────────────▼─┐  ┌──▼────────────┐
           │  Retriever    │  │  Generator     │
           │  (8 strategies│  │  (LangChain    │
           │  + rewrite)   │  │   LLM chain)   │
           └──────┬────────┘  └───────────────┘
                  │
        ┌─────────▼──────────┐
        │    Vector Store     │
        │  (ChromaDB +        │
        │   HuggingFace       │
        │   embeddings)       │
        └─────────────────────┘
```

### Why these choices

| Choice | Over | Because |
|---|---|---|
| **LangChain abstractions** (splitters, loaders, retrievers, LCEL chains) | Hand-rolled chunking/retrieval code | Every strategy shares one interface, so swapping strategies in the UI is a config change, not a rewrite — the whole point of a comparison lab |
| **ChromaDB** | Weaviate, FAISS | Zero-infra local persistence for a project meant to run on a laptop; Weaviate support is on the roadmap as a switchable backend for comparing vector store behavior too |
| **HuggingFace local embeddings (default)** | OpenAI embeddings | No API cost or key required to start experimenting; OpenAI/Anthropic remain available as swappable providers for cloud comparison |
| **Ollama as default LLM provider** | Cloud-only (OpenAI/Anthropic) | Keeps the "ingest → compare strategies" loop free and offline by default; cloud providers are one `.env` change away |
| **Streamlit UI** | Tkinter, custom web app | Fast to iterate on for a sidebar full of strategy toggles; state lives in `st.session_state` so switching strategies mid-conversation is cheap |

## Components

| File | Purpose |
|---|---|
| `raglab/rag_system.py` | Main orchestrator + CLI entry point |
| `raglab/config.py` | All configuration (reads from `.env`) |
| `gui_app.py` | Streamlit UI (chat interface + upload) |
| `raglab/ingestion/data_loader.py` | JSON/CSV ingestion |
| `raglab/ingestion/document_parser.py` | PDF, DOCX, TXT parsing via LangChain loaders |
| `raglab/ingestion/chunking.py` | 10 text splitting strategies |
| `raglab/retrieval/vector_store.py` | ChromaDB wrapper |
| `raglab/retrieval/retriever.py` | 8 retrieval strategies |
| `raglab/generation/generator.py` | LLM answer generation with multi-turn history |
| `raglab/retrieval/hybrid_search.py` | BM25 + EnsembleRetriever logic |
| `mcp_server.py` | MCP server exposing RAG as tools |
| `raglab/utils.py` | Logging, timers, retry decorators |
| `raglab/models.py` | Pydantic models for validation and response consistency |

## Configuration

All configuration is managed through environment variables in `.env`:

```bash
# LLM
LLM_PROVIDER=ollama            # openai | anthropic | ollama
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.7
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
COLLECTION_NAME=documents
# PERSIST_DIRECTORY is derived from the project root (vector_db/) and isn't env-configurable

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=semantic     # see Chunking Strategies table

# Retrieval
RETRIEVAL_STRATEGY=semantic    # see Retrieval Strategies table
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
SEMANTIC_WEIGHT=0.7            # hybrid only

# Advanced retrieval
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
PARENT_CHUNK_SIZE=1024
```

### Key Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | LLM provider | `ollama` |
| `LLM_MODEL` | Model name | `llama3.2` |
| `LLM_TEMPERATURE` | Generation temperature (0–1) | `0.7` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `COLLECTION_NAME` | Default collection | `documents` |
| `CHUNK_SIZE` | Chunk size in characters | `512` |
| `CHUNKING_STRATEGY` | Chunking method | `semantic` |
| `RETRIEVAL_STRATEGY` | Retrieval method | `semantic` |
| `TOP_K_RESULTS` | Chunks retrieved per query | `5` |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | `0.7` |
| `SEMANTIC_WEIGHT` | Hybrid search vector weight | `0.7` |
| `RERANKER_MODEL` | Cross-encoder for reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `PARENT_CHUNK_SIZE` | Parent chunk size (parent-child) | `1024` |

## Python API

```python
from raglab.rag_system import RAGSystem

rag = RAGSystem()

# Ingest a file
rag.ingest_data("docs/report.json", source_type="json")

# One-shot query
response = rag.query("What does the report say about Q4?")
print(response["answer_with_citations"])

# Multi-turn query
history = [
    {"role": "user", "content": "What is the total revenue?"},
    {"role": "assistant", "content": "Total revenue was $10M."},
]
response = rag.query(
    "How does that compare to Q3?",
    conversation_history=history,
    retrieval_strategy="hybrid",
    top_k=5,
)
```

## MCP Integration

Expose the RAG system as MCP tools for Claude Desktop and Claude Code.

See [MCP_SETUP.md](MCP_SETUP.md) for full setup instructions.

```bash
uv run python mcp_server.py
```

Available MCP tools: `ingest_data`, `ingest_document`, `query`, `search_documents`, `list_collections`, `get_collection_stats`, `delete_collection`, `clear_collection`, `get_system_stats`

## Testing

```bash
uv run pytest -v
uv run pytest --cov=raglab --cov-report=html
```

Tests are isolated from your working tree: `tests/conftest.py` redirects the vector store,
BM25 indexes, and log file into a temp directory, so running the suite never touches your
real `vector_db/`.

## CI

`.github/workflows/ci.yml` runs on every pull request and push to `main`: `ruff check`,
`ruff format --check`, and the test suite with a coverage gate (`--cov-fail-under=55`).
On pushes to `main` it also builds the Docker image as a validation step.

## Releasing

Releases are cut by pushing a version tag. A GitHub Actions workflow runs the tests,
publishes the multi-arch Docker image to `ghcr.io/kennexcorp/raglab`, and creates a GitHub
Release with auto-generated notes. The repo-level [`CHANGELOG.md`](CHANGELOG.md) is
regenerated from conventional commits with [git-cliff](https://git-cliff.org).

```bash
# 1. bump the version in pyproject.toml (must match the tag), commit it
# 2. regenerate and commit the changelog for the new version:
uv run git-cliff --tag vX.Y.Z -o CHANGELOG.md
# 3. tag and push — the workflow does the rest:
git tag vX.Y.Z
git push origin vX.Y.Z
```

The release workflow fails fast if the tag does not match the `version` in
`pyproject.toml`. Pre-release tags (`v1.2.3-rc1`) are marked as prereleases and publish
only the exact version tag, not the `{{major}}.{{minor}}` line.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OPENAI_API_KEY is required` | Add the key to `.env`, or switch `LLM_PROVIDER=ollama` for local inference |
| `ModuleNotFoundError: No module named 'chromadb'` | Run `uv sync` to ensure all packages are installed |
| `No relevant context found` | Ingest documents first. For reranking/HyDE strategies, lower `SIMILARITY_THRESHOLD` or switch strategies if the collection is small |
| Stale Streamlit cache after code changes | Restart the Streamlit server; `@st.cache_resource` persists across reruns |
| SpaCy / NLTK strategy errors | Run `uv run python -m spacy download en_core_web_sm` and `uv run python -c "import nltk; nltk.download('punkt')"` |

## Roadmap

- [ ] Weaviate vector store — switchable with ChromaDB via config, to compare vector store behavior alongside chunking/retrieval strategy
- [ ] Additional chunking methods (further sentence-based, structure-aware variants)
- [ ] Reranking refinements — expand cross-encoder reranker options
- [ ] Prompt templates with intent classification — route queries to the right collection/prompt automatically

## License

MIT License
