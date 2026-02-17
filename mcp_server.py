#!/usr/bin/env python3
"""
MCP Server for RAG System.
Exposes RAG functionality as MCP tools for integration with other applications.
"""

import asyncio
import logging
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from rag_system import RAGSystem
from config import Config
from utils import setup_logging


# Initialize RAG system
logger = setup_logging("INFO")
rag_system = RAGSystem(log_level="INFO")

# Create MCP server
app = Server("rag-dashboard-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG tools."""
    return [
        Tool(
            name="ingest_data",
            description="Ingest dashboard data into the RAG system from a JSON or CSV file. This loads the data, chunks it, and stores it in the vector database for querying.",
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
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="query_dashboard",
            description="Query the dashboard data using natural language. The system will retrieve relevant context and generate an AI-powered answer with source citations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the dashboard data",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of context chunks to retrieve (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="get_system_stats",
            description="Get statistics about the RAG system including document count, model information, and configuration.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="clear_data",
            description="Clear all data from the vector store. Use with caution as this will delete all ingested documents.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="search_documents",
            description="Search for relevant documents without generating an answer. Returns raw search results with similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""

    try:
        if name == "ingest_data":
            file_path = arguments.get("file_path")
            source_type = arguments.get("source_type", "json")

            num_chunks = rag_system.ingest_data(
                source=file_path, source_type=source_type
            )

            return [
                TextContent(
                    type="text",
                    text=f"Successfully ingested {num_chunks} chunks from {file_path}",
                )
            ]

        elif name == "query_dashboard":
            question = arguments.get("question")
            top_k = arguments.get("top_k", 5)

            response = rag_system.query(question, top_k=top_k)

            # Format response
            answer = response.get("answer_with_citations", response.get("answer", ""))
            num_sources = response.get("num_sources", 0)
            model = response.get("model", "N/A")
            tokens = response.get("tokens_used", "N/A")

            result = f"""Question: {question}

Answer:
{answer}

Metadata:
- Sources used: {num_sources}
- Model: {model}
- Tokens: {tokens}
"""

            return [TextContent(type="text", text=result)]

        elif name == "get_system_stats":
            stats = rag_system.get_stats()

            result = f"""RAG System Statistics:

Vector Store:
- Collection: {stats['vector_store']['collection_name']}
- Documents: {stats['vector_store']['document_count']}
- Embedding Model: {stats['vector_store']['embedding_model']}

Configuration:
- LLM Provider: {stats['config']['llm_provider']}
- LLM Model: {stats['config']['llm_model']}
- Embedding Model: {stats['config']['embedding_model']}
- Chunk Size: {stats['config']['chunk_size']} tokens
- Top K Results: {stats['config']['top_k']}
"""

            return [TextContent(type="text", text=result)]

        elif name == "clear_data":
            rag_system.clear_data()
            return [TextContent(type="text", text="All data cleared from vector store")]

        elif name == "search_documents":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)

            results = rag_system.retriever.retrieve(query, top_k=top_k)

            if not results:
                return [TextContent(type="text", text="No results found")]

            result_text = f"Found {len(results)} results for: '{query}'\n\n"

            for i, result in enumerate(results, 1):
                text = (
                    result["text"][:200] + "..."
                    if len(result["text"]) > 200
                    else result["text"]
                )
                similarity = result.get("similarity_score", "N/A")
                metadata = result.get("metadata", {})

                result_text += f"Result {i}:\n"
                result_text += f"Text: {text}\n"
                result_text += f"Similarity: {similarity}\n"
                result_text += f"Metadata: {metadata}\n\n"

            return [TextContent(type="text", text=result_text)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting RAG MCP Server")
    logger.info(Config.display_config())

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
