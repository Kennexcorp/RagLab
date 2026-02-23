# RAG System for Organization Dashboard

A production-ready Retrieval Augmented Generation (RAG) system that enables natural language querying of organization dashboard data using semantic search and LLM-powered answer generation.

## Features

- 📊 **Multi-source Data Ingestion**: Load data from JSON, CSV, or API endpoints
- 🔍 **Hybrid Search**: Combines semantic vector search (70%) + BM25 keyword search (30%) for optimal retrieval
- 🤖 **AI-Powered Answers**: Generate contextual answers with source citations
- 🎯 **Multiple LLM Support**: OpenAI, Anthropic Claude, or local Ollama models
- ⚡ **High Performance**: Optimized retrieval with ChromaDB vector store
- 📝 **Source Citations**: Answers include references to source data
- 🛠️ **CLI & Notebook**: Use via command line or interactive Jupyter notebook
- 🔌 **MCP Integration**: Expose as MCP tools for integration with AI assistants

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd "Retrieval Augmented Generation (RAG)"

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo
```

### 3. Ingest Data

```bash
# Ingest sample dashboard data
python rag_system.py --ingest sample_data.json --source-type json
```

### 4. Query the System

```bash
# Ask a question
python rag_system.py --query "How is our revenue performing?"

# Interactive mode
python rag_system.py --interactive
```

## Usage Examples

### Command Line Interface

```bash
# Ingest data
python rag_system.py --ingest sample_data.json

# Query with custom top-k
python rag_system.py --query "What are our user engagement metrics?" --top-k 3

# View system statistics
python rag_system.py --stats

# Clear all data
python rag_system.py --clear
```

### Python API

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# Ingest data
rag.ingest_data("sample_data.json", source_type="json")

# Query
response = rag.query("How is our revenue performing?")
print(response['answer_with_citations'])
```

### Jupyter Notebook

Open `index.ipynb` for an interactive walkthrough with examples and visualizations.

## Architecture

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         RAG System                  │
│  ┌──────────────────────────────┐  │
│  │  1. Retriever                │  │
│  │     - Semantic Search        │  │
│  │     - Context Building       │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  2. Generator                │  │
│  │     - Prompt Engineering     │  │
│  │     - LLM Integration        │  │
│  └──────────┬───────────────────┘  │
└─────────────┼───────────────────────┘
              │
              ▼
       ┌──────────────┐
       │   Answer     │
       │ + Citations  │
       └──────────────┘
```

## Components

- **`config.py`**: Centralized configuration management
- **`data_loader.py`**: Load and process data from various sources
- **`chunking.py`**: Split documents into semantic chunks
- **`vector_store.py`**: Vector database operations with ChromaDB
- **`retriever.py`**: Semantic search and context retrieval
- **`generator.py`**: LLM integration for answer generation
- **`rag_system.py`**: Main orchestrator and CLI
- **`utils.py`**: Logging, monitoring, and helper functions

## Configuration

All configuration is managed through environment variables in `.env`:

```bash
# LLM Configuration
LLM_PROVIDER=openai          # Options: openai, anthropic, ollama
LLM_MODEL=gpt-3.5-turbo      # Model name
OPENAI_API_KEY=your_key_here

# Embedding Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Sentence-transformers model

# Vector Store
COLLECTION_NAME=dashboard_data
PERSIST_DIRECTORY=./chroma_db

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=semantic

# Hybrid Search (Default)
USE_HYBRID_SEARCH=true       # Enable hybrid search
SEMANTIC_WEIGHT=0.7          # 70% weight for semantic search
KEYWORD_WEIGHT=0.3           # 30% weight for keyword (BM25) search
BM25_K1=1.5                  # BM25 term frequency saturation
BM25_B=0.75                  # BM25 length normalization

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Hybrid Search Explained

The system uses **hybrid search** by default, combining:

1. **Semantic Search (70%)**: Vector embeddings capture meaning and context
2. **Keyword Search (30%)**: BM25 algorithm for exact term matching

This provides the best of both worlds:
- Semantic understanding for conceptual queries
- Keyword precision for specific terms, dates, and IDs

You can adjust the weights or disable hybrid search:
```bash
# Use semantic search only
USE_HYBRID_SEARCH=false

# Adjust weights (must sum to 1.0)
SEMANTIC_WEIGHT=0.8
KEYWORD_WEIGHT=0.2
```

## Configuration Options

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (openai/anthropic/ollama) | openai |
| `LLM_MODEL` | Model name | gpt-3.5-turbo |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Chunk size in tokens | 512 |
| `TOP_K_RESULTS` | Number of results to retrieve | 5 |
| `CHUNKING_STRATEGY` | Chunking method (semantic/fixed) | semantic |

## Sample Data

The included `sample_data.json` contains 10 dashboard records covering:
- Finance & Revenue
- User Analytics
- Customer Satisfaction
- Sales Pipeline
- Product Adoption
- Marketing Performance
- Infrastructure Metrics
- Employee Engagement
- Support Tickets
- Competitive Analysis

## Performance

Typical query performance:
- **Retrieval**: < 500ms
- **Generation**: 1-3s (depending on LLM)
- **Total**: < 5s end-to-end

## Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Troubleshooting

### API Key Issues
```
Error: OPENAI_API_KEY is required
```
**Solution**: Add your API key to `.env` file

### Import Errors
```
ModuleNotFoundError: No module named 'chromadb'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Empty Results
```
No relevant context found
```
**Solution**: Ensure data is ingested first with `--ingest` command

## MCP Integration

Expose your RAG system as MCP (Model Context Protocol) tools for integration with AI assistants like Claude Desktop.

See [MCP_SETUP.md](MCP_SETUP.md) for detailed setup instructions.

**Quick setup:**
```bash
# Install MCP dependency
pip install mcp

# Run the MCP server
python mcp_server.py
```

Available MCP tools:
- `ingest_data` - Load dashboard data
- `query_dashboard` - Ask questions
- `search_documents` - Search without generation
- `get_system_stats` - View system info
- `clear_data` - Clear all data

## Advanced Usage

### Custom Data Sources

```python
from data_loader import DataLoader

loader = DataLoader()

# Load from CSV
documents = loader.load_and_process(
    "data.csv",
    source_type="csv",
    text_fields=["title", "description"],
    metadata_fields=["date", "category"]
)
```

### Direct Component Access

```python
from vector_store import VectorStore
from retriever import Retriever

# Initialize components
vector_store = VectorStore()
retriever = Retriever(vector_store)

# Retrieve without generation
results = retriever.retrieve("revenue metrics", top_k=5)
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please open a GitHub issue.
