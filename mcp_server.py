#!/usr/bin/env python3
"""
MCP Server for RAG System.
Exposes document ingestion, querying, and collection management as MCP tools.
"""

import asyncio
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag_system import RAGSystem
from config import Config
from utils import setup_logging


logger = setup_logging("INFO")
rag_system = RAGSystem(log_level="INFO")

app = Server("rag-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG tools."""
    return [
        Tool(
            name="ingest_data",
            description=(
                "Ingest a JSON or CSV file into the RAG system. "
                "The file is loaded, chunked, and stored in the vector database."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the data file (JSON or CSV)",
                    },
                    "source_type": {
                        "type": "string",
                        "enum": ["json", "csv"],
                        "description": "Type of data source",
                        "default": "json",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Target collection name (defaults to the system default)",
                        "default": "",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="ingest_document",
            description=(
                "Ingest a plain-text document with rich metadata into the RAG system. "
                "Use this when you have already extracted text from a file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Full plain-text content of the document",
                    },
                    "title": {
                        "type": "string",
                        "description": "Human-readable document title",
                    },
                    "category": {
                        "type": "string",
                        "description": "Document category, e.g. 'finance', 'hr', 'legal'",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source label, e.g. original filename",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional freeform description",
                        "default": "",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags, e.g. 'Q4,2026,revenue'",
                        "default": "",
                    },
                    "author": {
                        "type": "string",
                        "description": "Optional author name",
                        "default": "",
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Target collection name (defaults to the system default)",
                        "default": "",
                    },
                },
                "required": ["text", "title", "category", "source"],
            },
        ),
        Tool(
            name="query",
            description=(
                "Query the knowledge base using natural language. "
                "Retrieves relevant context and returns an AI-generated answer with source citations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of context chunks to retrieve (default: 5)",
                        "default": 5,
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection to search (defaults to the system default)",
                        "default": "",
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="search_documents",
            description=(
                "Search for relevant documents without generating an answer. "
                "Returns raw results with similarity scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "Collection to search (defaults to the system default)",
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_collections",
            description="List all collections in the vector store.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_collection_stats",
            description="Get statistics for a collection including document count and model info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Collection name (defaults to the system default)",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="delete_collection",
            description="Permanently delete a collection and all its documents. Use with caution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to delete",
                    },
                },
                "required": ["collection_name"],
            },
        ),
        Tool(
            name="clear_collection",
            description="Remove all documents from a collection without deleting the collection itself.",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Collection to clear (defaults to the system default)",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="get_system_stats",
            description="Get overall system statistics including configuration and default collection info.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    try:
        col = arguments.get("collection_name") or None  # treat empty string as None

        if name == "ingest_data":
            file_path = arguments.get("file_path")
            source_type = arguments.get("source_type", "json")
            num_chunks = rag_system.ingest_data(
                source=file_path, source_type=source_type, collection_name=col
            )
            return [TextContent(
                type="text",
                text=f"Ingested {num_chunks} chunks from '{file_path}' into collection '{col or Config.COLLECTION_NAME}'.",
            )]

        elif name == "ingest_document":
            num_chunks = rag_system.ingest_document_with_metadata(
                text=arguments.get("text"),
                title=arguments.get("title"),
                category=arguments.get("category"),
                source=arguments.get("source"),
                description=arguments.get("description", ""),
                tags=arguments.get("tags", ""),
                author=arguments.get("author", ""),
                collection_name=col,
            )
            return [TextContent(
                type="text",
                text=f"Ingested '{arguments.get('title')}': {num_chunks} chunks stored in '{col or Config.COLLECTION_NAME}'.",
            )]

        elif name == "query":
            question = arguments.get("question")
            top_k = arguments.get("top_k", 5)
            response = rag_system.query(question, top_k=top_k, collection_name=col)

            answer = response.get("answer_with_citations", response.get("answer", ""))
            result = (
                f"Question: {question}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Sources used: {response.get('num_sources', 0)} | "
                f"Model: {response.get('model', 'N/A')} | "
                f"Tokens: {response.get('tokens_used', 'N/A')}"
            )
            return [TextContent(type="text", text=result)]

        elif name == "search_documents":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            _, ret = rag_system._get_or_create_collection(col or Config.COLLECTION_NAME)
            results = ret.retrieve(query, top_k=top_k)

            if not results:
                return [TextContent(type="text", text="No results found.")]

            result_text = f"Found {len(results)} results for: '{query}'\n\n"
            for i, result in enumerate(results, 1):
                text = result["text"][:200] + ("..." if len(result["text"]) > 200 else "")
                result_text += (
                    f"Result {i}:\n"
                    f"  Text: {text}\n"
                    f"  Similarity: {result.get('similarity_score', 'N/A')}\n"
                    f"  Metadata: {result.get('metadata', {})}\n\n"
                )
            return [TextContent(type="text", text=result_text)]

        elif name == "list_collections":
            collections = rag_system.list_collections()
            if collections:
                return [TextContent(type="text", text="Collections:\n" + "\n".join(f"  - {c}" for c in collections))]
            return [TextContent(type="text", text="No collections found.")]

        elif name == "get_collection_stats":
            stats = rag_system.get_collection_stats(col)
            result = (
                f"Collection: {stats['collection_name']}\n"
                f"Documents: {stats['document_count']}\n"
                f"Embedding Model: {stats['embedding_model']}"
            )
            return [TextContent(type="text", text=result)]

        elif name == "delete_collection":
            collection_name = arguments.get("collection_name")
            rag_system.delete_collection(collection_name)
            return [TextContent(type="text", text=f"Collection '{collection_name}' deleted.")]

        elif name == "clear_collection":
            rag_system.clear_data(collection_name=col)
            return [TextContent(type="text", text=f"Collection '{col or Config.COLLECTION_NAME}' cleared.")]

        elif name == "get_system_stats":
            stats = rag_system.get_stats()
            result = (
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
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error executing tool '{name}': {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting RAG MCP Server")
    logger.info(Config.display_config())

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
