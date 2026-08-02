#!/usr/bin/env python3
"""
MCP Server for RAG System.
Exposes document ingestion, querying, and collection management as MCP tools.
"""

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field

from raglab.config import Config
from raglab.rag_system import RAGSystem
from raglab.utils import setup_logging

logger = setup_logging("INFO")
rag_system = RAGSystem(log_level="INFO")

app = Server("rag-server")


# ---------------------------------------------------------------------------
# Tool argument models — source of truth for both the MCP inputSchema and
# runtime validation, so the two can never drift apart.
# ---------------------------------------------------------------------------


class IngestDataArgs(BaseModel):
    file_path: str = Field(min_length=1, description="Path to the data file (JSON or CSV)")
    source_type: Literal["json", "csv"] = Field("json", description="Type of data source")
    collection_name: str = Field(
        "", description="Target collection name (defaults to the system default)"
    )


class IngestDocumentArgs(BaseModel):
    text: str = Field(min_length=1, description="Full plain-text content of the document")
    title: str = Field(min_length=1, description="Human-readable document title")
    category: str = Field(
        min_length=1, description="Document category, e.g. 'finance', 'hr', 'legal'"
    )
    source: str = Field(min_length=1, description="Source label, e.g. original filename")
    description: str = Field("", description="Optional freeform description")
    tags: str = Field("", description="Comma-separated tags, e.g. 'Q4,2026,revenue'")
    author: str = Field("", description="Optional author name")
    collection_name: str = Field(
        "", description="Target collection name (defaults to the system default)"
    )


class QueryArgs(BaseModel):
    question: str = Field(min_length=1, description="Natural language question")
    top_k: int = Field(5, gt=0, description="Number of context chunks to retrieve (default: 5)")
    collection_name: str = Field(
        "", description="Collection to search (defaults to the system default)"
    )


class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=1, description="Search query")
    top_k: int = Field(5, gt=0, description="Number of results to return (default: 5)")
    collection_name: str = Field(
        "", description="Collection to search (defaults to the system default)"
    )


class DeleteCollectionArgs(BaseModel):
    collection_name: str = Field(min_length=1, description="Name of the collection to delete")


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
            inputSchema=IngestDataArgs.model_json_schema(),
        ),
        Tool(
            name="ingest_document",
            description=(
                "Ingest a plain-text document with rich metadata into the RAG system. "
                "Use this when you have already extracted text from a file."
            ),
            inputSchema=IngestDocumentArgs.model_json_schema(),
        ),
        Tool(
            name="query",
            description=(
                "Query the knowledge base using natural language. "
                "Retrieves relevant context and returns an AI-generated answer with "
                "source citations."
            ),
            inputSchema=QueryArgs.model_json_schema(),
        ),
        Tool(
            name="search_documents",
            description=(
                "Search for relevant documents without generating an answer. "
                "Returns raw results with similarity scores."
            ),
            inputSchema=SearchDocumentsArgs.model_json_schema(),
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
            inputSchema=DeleteCollectionArgs.model_json_schema(),
        ),
        Tool(
            name="clear_collection",
            description=(
                "Remove all documents from a collection without deleting the collection itself."
            ),
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
            description=(
                "Get overall system statistics including configuration and default collection info."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    try:
        col = arguments.get("collection_name") or None  # treat empty string as None

        if name == "ingest_data":
            args = IngestDataArgs.model_validate(arguments)
            num_chunks = rag_system.ingest_data(
                source=args.file_path, source_type=args.source_type, collection_name=col
            )
            target_collection = col or Config.COLLECTION_NAME
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Ingested {num_chunks} chunks from '{args.file_path}' "
                        f"into collection '{target_collection}'."
                    ),
                )
            ]

        elif name == "ingest_document":
            args = IngestDocumentArgs.model_validate(arguments)
            num_chunks = rag_system.ingest_document_with_metadata(
                text=args.text,
                title=args.title,
                category=args.category,
                source=args.source,
                description=args.description,
                tags=args.tags,
                author=args.author,
                collection_name=col,
            )
            target_collection = col or Config.COLLECTION_NAME
            return [
                TextContent(
                    type="text",
                    text=(
                        f"Ingested '{args.title}': {num_chunks} chunks "
                        f"stored in '{target_collection}'."
                    ),
                )
            ]

        elif name == "query":
            args = QueryArgs.model_validate(arguments)
            response = rag_system.query(args.question, top_k=args.top_k, collection_name=col)

            answer = response.get("answer_with_citations", response.get("answer", ""))
            result = (
                f"Question: {args.question}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Sources used: {response.get('num_sources', 0)} | "
                f"Model: {response.get('model', 'N/A')} | "
                f"Tokens: {response.get('tokens_used', 'N/A')}"
            )
            return [TextContent(type="text", text=result)]

        elif name == "search_documents":
            args = SearchDocumentsArgs.model_validate(arguments)
            _, ret = rag_system._get_or_create_collection(col or Config.COLLECTION_NAME)
            results = ret.retrieve(args.query, top_k=args.top_k)

            if not results:
                return [TextContent(type="text", text="No results found.")]

            result_text = f"Found {len(results)} results for: '{args.query}'\n\n"
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
                return [
                    TextContent(
                        type="text",
                        text="Collections:\n" + "\n".join(f"  - {c}" for c in collections),
                    )
                ]
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
            args = DeleteCollectionArgs.model_validate(arguments)
            rag_system.delete_collection(args.collection_name)
            return [TextContent(type="text", text=f"Collection '{args.collection_name}' deleted.")]

        elif name == "clear_collection":
            rag_system.clear_data(collection_name=col)
            return [
                TextContent(
                    type="text",
                    text=f"Collection '{col or Config.COLLECTION_NAME}' cleared.",
                )
            ]

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
