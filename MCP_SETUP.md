# MCP Server Setup for RAG System

This guide explains how to expose your RAG system as MCP (Model Context Protocol) tools.

## What is MCP?

MCP (Model Context Protocol) is a standardized protocol that allows applications to expose tools and resources that can be used by AI assistants and other applications. By exposing your RAG system as MCP tools, you can integrate it with various MCP-compatible clients.

## Available MCP Tools

The RAG MCP server exposes the following tools:

### 1. `ingest_data`
Ingest dashboard data into the RAG system from a JSON or CSV file.

**Parameters:**
- `file_path` (required): Path to the data file
- `source_type` (optional): "json" or "csv" (default: "json")

**Example:**
```json
{
  "file_path": "sample_data.json",
  "source_type": "json"
}
```

### 2. `query_dashboard`
Query the dashboard data using natural language.

**Parameters:**
- `question` (required): Natural language question
- `top_k` (optional): Number of context chunks to retrieve (default: 5)

**Example:**
```json
{
  "question": "What is our Q4 revenue?",
  "top_k": 3
}
```

### 3. `search_documents`
Search for relevant documents without generating an answer.

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Number of results (default: 5)

**Example:**
```json
{
  "query": "revenue metrics",
  "top_k": 5
}
```

### 4. `ingest_keytable`
Ingest a keytable JSON API response into the RAG system. Each row in the series tree becomes one document. Accepts both the raw full API payload and the slim pre-filtered format — the loader identifies the relevant columns automatically.

**Parameters:**
- `file_path` (required): Path to the keytable JSON file

**Example:**
```json
{
  "file_path": "/path/to/keytable-raw.json"
}
```

### 5. `get_system_stats`
Get statistics about the RAG system.

**Parameters:** None

### 6. `clear_data`
Clear all data from the vector store.

**Parameters:** None

## Setup Instructions

### 1. Install MCP Dependency

```bash
pip install mcp
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Make sure your `.env` file is set up with your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Test the MCP Server

Run the server directly to test:

```bash
python mcp_server.py
```

The server will start and listen for MCP protocol messages on stdin/stdout.

### 4. Register with MCP Client

To use the RAG server with an MCP-compatible client (like Claude Desktop), add the configuration to your MCP settings:

**For Claude Desktop:**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag-dashboard": {
      "command": "python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)"
      }
    }
  }
}
```

**Note:** Update the path to match your actual installation directory.

**For Claude Code (CLI):**

Add the server at project scope using the CLI:

```bash
claude mcp add rag-dashboard \
  /Users/sly/Documents/MachineLearning/Retrieval\ Augmented\ Generation\ \(RAG\)/.conda/bin/python \
  /Users/sly/Documents/MachineLearning/Retrieval\ Augmented\ Generation\ \(RAG\)/mcp_server.py
```

Or add it manually as a project-scoped server by creating `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "rag-dashboard": {
      "command": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.conda/bin/python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ]
    }
  }
}
```

To add it globally (available across all projects), add the same `mcpServers` block to `~/.claude/settings.json`.

Verify the server is connected:
```bash
claude mcp list
```

### 5. Activate Conda Environment (if using)

The simplest approach is to point directly to the conda env's Python binary — no activation step needed:

```json
{
  "mcpServers": {
    "rag-dashboard": {
      "command": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.conda/bin/python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ]
    }
  }
}
```

This works for both Claude Desktop and Claude Code (`.mcp.json`). The project-local `.conda/bin/python` already has all dependencies installed, so no `conda run` wrapper is needed.

## Usage Examples

Once configured, you can use the tools from any MCP-compatible client:

### Example 1: Ingest Data
```
Use the ingest_data tool to load sample_data.json
```

### Example 2: Query Dashboard
```
Use the query_dashboard tool to ask: "What are our top performing metrics?"
```

### Example 3: Search Documents
```
Use the search_documents tool to find documents about "revenue"
```

### Example 4: Get Stats
```
Use the get_system_stats tool to see system information
```

## Troubleshooting

### Server Not Starting

**Issue:** Server fails to start
**Solution:** Check that all dependencies are installed and your `.env` file is configured

### API Key Errors

**Issue:** "API key required" errors
**Solution:** Ensure your `.env` file contains valid API keys for your chosen LLM provider

### Path Issues

**Issue:** "Module not found" errors
**Solution:** Verify the PYTHONPATH in your MCP config points to the correct directory

### No Data Found

**Issue:** Queries return "No relevant context found"
**Solution:** Make sure you've ingested data first using the `ingest_data` tool

## Advanced Configuration

### Custom Models

You can override any `.env` setting per-server via `env` in the MCP config:

```json
{
  "mcpServers": {
    "rag-dashboard": {
      "command": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.conda/bin/python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ],
      "env": {
        "LLM_PROVIDER": "anthropic",
        "LLM_MODEL": "claude-sonnet-4-6",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "TOP_K_RESULTS": "5"
      }
    }
  }
}
```

### Multiple Instances

Run multiple RAG servers pointing at different ChromaDB collections:

```json
{
  "mcpServers": {
    "rag-gold-prod": {
      "command": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.conda/bin/python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ],
      "env": {
        "COLLECTION_NAME": "gold_production"
      }
    },
    "rag-gold-dev": {
      "command": "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/.conda/bin/python",
      "args": [
        "/Users/sly/Documents/MachineLearning/Retrieval Augmented Generation (RAG)/mcp_server.py"
      ],
      "env": {
        "COLLECTION_NAME": "gold_development"
      }
    }
  }
}
```

## Security Considerations

- **API Keys:** Never commit `.env` files with real API keys
- **Data Access:** The MCP server has full access to your RAG system
- **Clear Data:** The `clear_data` tool permanently deletes all ingested data
- **File Paths:** Validate file paths in production to prevent unauthorized access

## Next Steps

1. Ingest your actual dashboard data
2. Test queries through the MCP interface
3. Integrate with your preferred MCP client
4. Monitor performance and adjust configuration as needed

For more information about the RAG system itself, see the main [README.md](README.md).
