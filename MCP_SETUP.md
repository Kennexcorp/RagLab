# MCP Server Setup for RAG System

This guide explains how to expose your RAG system as MCP (Model Context Protocol) tools for use with Claude Desktop, Claude Code, and other MCP-compatible clients.

## Available MCP Tools

### `ingest_data`
Load a JSON or CSV file into the RAG system. The file is chunked and stored in the vector database.

**Parameters:**
- `file_path` (required): Path to the data file
- `source_type` (optional): `"json"` or `"csv"` (default: `"json"`)
- `collection_name` (optional): Target collection (defaults to system default)

**Example:**
```json
{ "file_path": "docs/report.json", "source_type": "json" }
```

---

### `ingest_document`
Ingest a plain-text document with structured metadata.

**Parameters:**
- `text` (required): Full plain-text content
- `title` (required): Human-readable document title
- `category` (required): Document category (e.g. `"finance"`, `"legal"`)
- `source` (required): Source label or original filename
- `description` (optional): Freeform description
- `tags` (optional): Comma-separated tags, e.g. `"Q4,2026,revenue"`
- `author` (optional): Author name
- `collection_name` (optional): Target collection

**Example:**
```json
{
  "text": "Q4 revenue reached $10M, up 25% year-over-year...",
  "title": "Q4 2025 Finance Report",
  "category": "finance",
  "source": "q4-report.pdf",
  "tags": "Q4,2025,revenue"
}
```

---

### `query`
Query the knowledge base using natural language. Returns an AI-generated answer with source citations.

**Parameters:**
- `question` (required): Natural language question
- `top_k` (optional): Number of context chunks to retrieve (default: 5)
- `collection_name` (optional): Collection to search

**Example:**
```json
{ "question": "What was Q4 revenue?", "top_k": 3 }
```

---

### `search_documents`
Search for relevant documents without generating an answer. Returns raw results with similarity scores.

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Number of results (default: 5)
- `collection_name` (optional): Collection to search

**Example:**
```json
{ "query": "revenue metrics", "top_k": 5 }
```

---

### `list_collections`
List all collections in the vector store.

**Parameters:** None

---

### `get_collection_stats`
Get statistics for a collection: document count, embedding model.

**Parameters:**
- `collection_name` (optional): Collection name (defaults to system default)

---

### `delete_collection`
Permanently delete a collection and all its documents.

**Parameters:**
- `collection_name` (required): Name of the collection to delete

---

### `clear_collection`
Remove all documents from a collection without deleting the collection itself.

**Parameters:**
- `collection_name` (optional): Collection to clear (defaults to system default)

---

### `get_system_stats`
Get overall system statistics: configuration, default collection info, document count.

**Parameters:** None

---

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

Minimum `.env` for local Ollama use:
```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

### 3. Test the Server

```bash
uv run python mcp_server.py
```

The server starts and listens on stdin/stdout (MCP stdio transport).

---

## Register with MCP Clients

### Option A: Using `uv` (Recommended)
This approach leverages the `uv` command-line tool directly. It automatically handles virtual environment configuration and is highly portable.

#### Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)",
        "run",
        "mcp_server.py"
      ]
    }
  }
}
```

#### Claude Code (project-scoped)
Create `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)",
        "run",
        "mcp_server.py"
      ]
    }
  }
}
```

Or add via the CLI:
```bash
claude mcp add rag \
  uv \
  --directory "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)" \
  run mcp_server.py
```

---

### Option B: Using the `.venv` Python Binary
If you prefer not to call `uv` globally, you can point directly to the python interpreter created in the local virtual environment.

#### Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.venv/bin/python",
      "args": [
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ]
    }
  }
}
```

---

## Advanced Configuration

Override `.env` settings per-server via the `env` key in your MCP config:

```json
{
  "mcpServers": {
    "rag": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)",
        "run",
        "mcp_server.py"
      ],
      "env": {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-sonnet-4-6",
        "COLLECTION_NAME": "legal-docs"
      }
    }
  }
}
```

### Multiple Instances (multiple collections)

```json
{
  "mcpServers": {
    "rag-legal": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)",
        "run",
        "mcp_server.py"
      ],
      "env": { "COLLECTION_NAME": "legal" }
    },
    "rag-finance": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/sylvester/Documents/MachineLearning/Retrieval Augmented Generation (RAG)",
        "run",
        "mcp_server.py"
      ],
      "env": { "COLLECTION_NAME": "finance" }
    }
  }
}
```

---

## Troubleshooting

**Server fails to start** — Check all dependencies are installed and `.env` is configured.

**`API key required` errors** — Ensure `.env` has a valid key for your chosen `LLM_PROVIDER`, or switch to `ollama` for local inference.

**`Module not found` errors** — Verify the `command` path points to the virtual environment python (`.venv/bin/python`), or use `uv` as the command.

**`No relevant context found`** — Ingest documents first using `ingest_data` or `ingest_document`.

---

## Security

- Never commit `.env` files containing real API keys
- `delete_collection` and `clear_collection` are destructive — use with care
- Validate file paths in production to prevent unauthorized file access

---

For more on the RAG system, see [README.md](README.md).
