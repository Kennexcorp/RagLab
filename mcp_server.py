#!/usr/bin/env python3
"""
MCP Server for RAG System.
Exposes document ingestion, querying, and collection management as MCP tools.
"""

from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from raglab.config import Config
from raglab.rag_system import RAGSystem
from raglab.utils import setup_logging

logger = setup_logging("INFO")
rag_system = RAGSystem(log_level="INFO")

mcp = FastMCP("rag-server")


@mcp.tool(
    description=(
        "Ingest a JSON or CSV file into the RAG system. "
        "The file is loaded, chunked, and stored in the vector database."
    )
)
async def ingest_data(
    file_path: Annotated[
        str, Field(min_length=1, description="Path to the data file (JSON or CSV)")
    ],
    source_type: Annotated[
        Literal["json", "csv"], Field(description="Type of data source")
    ] = "json",
    collection_name: Annotated[
        str, Field(description="Target collection name (defaults to the system default)")
    ] = "",
) -> str:
    try:
        col = collection_name or None
        num_chunks = rag_system.ingest_data(
            source=file_path, source_type=source_type, collection_name=col
        )
        target_collection = col or Config.COLLECTION_NAME
        return (
            f"Ingested {num_chunks} chunks from '{file_path}' "
            f"into collection '{target_collection}'."
        )
    except Exception:
        logger.exception("Error executing tool 'ingest_data'")
        raise


@mcp.tool(
    description=(
        "Ingest a plain-text document with rich metadata into the RAG system. "
        "Use this when you have already extracted text from a file."
    )
)
async def ingest_document(
    text: Annotated[
        str, Field(min_length=1, description="Full plain-text content of the document")
    ],
    title: Annotated[str, Field(min_length=1, description="Human-readable document title")],
    category: Annotated[
        str, Field(min_length=1, description="Document category, e.g. 'finance', 'hr', 'legal'")
    ],
    source: Annotated[str, Field(min_length=1, description="Source label, e.g. original filename")],
    description: Annotated[str, Field(description="Optional freeform description")] = "",
    tags: Annotated[str, Field(description="Comma-separated tags, e.g. 'Q4,2026,revenue'")] = "",
    author: Annotated[str, Field(description="Optional author name")] = "",
    collection_name: Annotated[
        str, Field(description="Target collection name (defaults to the system default)")
    ] = "",
) -> str:
    try:
        col = collection_name or None
        num_chunks = rag_system.ingest_document_with_metadata(
            text=text,
            title=title,
            category=category,
            source=source,
            description=description,
            tags=tags,
            author=author,
            collection_name=col,
        )
        target_collection = col or Config.COLLECTION_NAME
        return f"Ingested '{title}': {num_chunks} chunks stored in '{target_collection}'."
    except Exception:
        logger.exception("Error executing tool 'ingest_document'")
        raise


@mcp.tool(
    description=(
        "Query the knowledge base using natural language. "
        "Retrieves relevant context and returns an AI-generated answer with source citations."
    )
)
async def query(
    question: Annotated[str, Field(min_length=1, description="Natural language question")],
    top_k: Annotated[
        int, Field(gt=0, description="Number of context chunks to retrieve (default: 5)")
    ] = 5,
    collection_name: Annotated[
        str, Field(description="Collection to search (defaults to the system default)")
    ] = "",
) -> str:
    try:
        col = collection_name or None
        response = rag_system.query(question, top_k=top_k, collection_name=col)
        answer = response.get("answer_with_citations", response.get("answer", ""))
        return (
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Sources used: {response.get('num_sources', 0)} | "
            f"Model: {response.get('model', 'N/A')} | "
            f"Tokens: {response.get('tokens_used', 'N/A')}"
        )
    except Exception:
        logger.exception("Error executing tool 'query'")
        raise


@mcp.tool(
    description=(
        "Search for relevant documents without generating an answer. "
        "Returns raw results with similarity scores."
    )
)
async def search_documents(
    query: Annotated[str, Field(min_length=1, description="Search query")],
    top_k: Annotated[int, Field(gt=0, description="Number of results to return (default: 5)")] = 5,
    collection_name: Annotated[
        str, Field(description="Collection to search (defaults to the system default)")
    ] = "",
) -> str:
    try:
        col = collection_name or None
        _, ret = rag_system._get_or_create_collection(col or Config.COLLECTION_NAME)
        results = ret.retrieve(query, top_k=top_k)

        if not results:
            return "No results found."

        result_text = f"Found {len(results)} results for: '{query}'\n\n"
        for i, result in enumerate(results, 1):
            text = result["text"][:200] + ("..." if len(result["text"]) > 200 else "")
            result_text += (
                f"Result {i}:\n"
                f"  Text: {text}\n"
                f"  Similarity: {result.get('similarity_score', 'N/A')}\n"
                f"  Metadata: {result.get('metadata', {})}\n\n"
            )
        return result_text
    except Exception:
        logger.exception("Error executing tool 'search_documents'")
        raise


@mcp.tool(description="List all collections in the vector store.")
async def list_collections() -> str:
    try:
        collections = rag_system.list_collections()
        if collections:
            return "Collections:\n" + "\n".join(f"  - {c}" for c in collections)
        return "No collections found."
    except Exception:
        logger.exception("Error executing tool 'list_collections'")
        raise


@mcp.tool(description="Get statistics for a collection including document count and model info.")
async def get_collection_stats(
    collection_name: Annotated[
        str, Field(description="Collection name (defaults to the system default)")
    ] = "",
) -> str:
    try:
        stats = rag_system.get_collection_stats(collection_name or None)
        return (
            f"Collection: {stats['collection_name']}\n"
            f"Documents: {stats['document_count']}\n"
            f"Embedding Model: {stats['embedding_model']}"
        )
    except Exception:
        logger.exception("Error executing tool 'get_collection_stats'")
        raise


@mcp.tool(description="Permanently delete a collection and all its documents. Use with caution.")
async def delete_collection(
    collection_name: Annotated[
        str, Field(min_length=1, description="Name of the collection to delete")
    ],
) -> str:
    try:
        rag_system.delete_collection(collection_name)
        return f"Collection '{collection_name}' deleted."
    except Exception:
        logger.exception("Error executing tool 'delete_collection'")
        raise


@mcp.tool(
    description="Remove all documents from a collection without deleting the collection itself."
)
async def clear_collection(
    collection_name: Annotated[
        str, Field(description="Collection to clear (defaults to the system default)")
    ] = "",
) -> str:
    try:
        col = collection_name or None
        rag_system.clear_data(collection_name=col)
        return f"Collection '{col or Config.COLLECTION_NAME}' cleared."
    except Exception:
        logger.exception("Error executing tool 'clear_collection'")
        raise


@mcp.tool(
    description=(
        "Get overall system statistics including configuration and default collection info."
    )
)
async def get_system_stats() -> str:
    try:
        stats = rag_system.get_stats()
        return (
            f"RAG System Statistics\n"
            f"---------------------\n"
            f"Default Collection: {stats['vector_store']['collection_name']}\n"
            f"Documents (default): {stats['vector_store']['document_count']}\n"
            f"Embedding Model: {stats['vector_store']['embedding_model']}\n"
            f"LLM Provider: {stats['config']['llm_provider']}\n"
            f"LLM Model: {stats['config']['llm_model']}\n"
            f"Chunk Size: {stats['config']['chunk_size']} chars\n"
            f"Top K: {stats['config']['top_k']}"
        )
    except Exception:
        logger.exception("Error executing tool 'get_system_stats'")
        raise


if __name__ == "__main__":
    logger.info("Starting RAG MCP Server")
    logger.info(Config.display_config())
    mcp.run()
