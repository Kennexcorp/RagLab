"""
Main RAG System orchestrator.
Coordinates all components to provide end-to-end document ingestion and question answering.
"""

import argparse
import os
from typing import Any

from chunking import TextChunker
from config import Config
from data_loader import DataLoader
from generator import Generator
from models import DocumentMetadata, QueryResponse
from retriever import Retriever
from utils import PerformanceMonitor, setup_logging
from vector_store import VectorStore


class RAGSystem:
    """Main RAG system that coordinates all components."""

    def __init__(self, log_level: str = "INFO"):
        """
        Initialize RAG system.

        Args:
            log_level: Logging level
        """
        _log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_system.log")
        self.logger = setup_logging(log_level, log_file=_log_file)
        self.log_level = log_level
        self.logger.info("Initializing RAG System")

        # Validate configuration
        if not Config.validate():
            raise ValueError("Configuration validation failed")

        # Shared components (collection-independent)
        self.data_loader = DataLoader(log_level=log_level)
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            strategy=Config.CHUNKING_STRATEGY,
            log_level=log_level,
        )
        self.performance_monitor = PerformanceMonitor()

        # Per-collection cache: {collection_name: (VectorStore, Retriever)}
        # VectorStore instances are cached to avoid reloading the embedding model.
        self._collection_cache: dict[str, tuple[VectorStore, Retriever]] = {}

        # Default collection — kept as named attributes for CLI backward compatibility
        self.vector_store = self._get_or_create_collection(Config.COLLECTION_NAME)[0]
        self.retriever = self._get_or_create_collection(Config.COLLECTION_NAME)[1]

        # Default generator (used by CLI; GUI creates per-query generators)
        self.generator = Generator(log_level=log_level)

        self.logger.info("RAG System initialized successfully")
        self.logger.info(Config.display_config())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(self, collection_name: str) -> tuple[VectorStore, Retriever]:
        """Return (VectorStore, Retriever) for the given collection, creating if needed."""
        if collection_name not in self._collection_cache:
            vs = VectorStore(collection_name=collection_name, log_level=self.log_level)
            ret = Retriever(vector_store=vs, log_level=self.log_level)
            self._collection_cache[collection_name] = (vs, ret)
        return self._collection_cache[collection_name]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _rewrite_query(
        self,
        question: str,
        conversation_history: list[dict[str, Any]],
        provider: str = None,
        model: str = None,
        api_key: str = None,
    ) -> str:
        """Rewrite a follow-up question into a standalone search query using the LLM."""
        from langchain_core.messages import HumanMessage

        gen = Generator(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=0,
            log_level=self.log_level,
        )
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in conversation_history[-4:]  # last 2 turns is sufficient context
        )
        prompt = (
            f"Given this conversation:\n{history_text}\n\n"
            f"Rewrite the follow-up question as a complete, self-contained search query "
            f"that can be understood without the conversation context. "
            f"Output only the rewritten question, nothing else.\n\n"
            f"Follow-up: {question}"
        )
        response = gen._llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()
        self.logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
        return rewritten

    def _make_chunker(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
    ):
        """Return a TextChunker with custom settings, or the shared default chunker."""
        if any([chunk_size, chunk_overlap, chunk_strategy]):
            return TextChunker(
                chunk_size=chunk_size or Config.CHUNK_SIZE,
                chunk_overlap=chunk_overlap or Config.CHUNK_OVERLAP,
                strategy=chunk_strategy or Config.CHUNKING_STRATEGY,
                log_level=self.log_level,
            )
        return self.chunker

    def ingest_data(
        self,
        source: str,
        source_type: str = "json",
        collection_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
    ) -> int:
        """
        Ingest data from a JSON or CSV file.

        Args:
            source:          Path to data file
            source_type:     'json' or 'csv'
            collection_name: Target collection (defaults to Config.COLLECTION_NAME)

        Returns:
            Number of chunks ingested
        """
        collection_name = collection_name or Config.COLLECTION_NAME
        vs, ret = self._get_or_create_collection(collection_name)

        self.logger.info(
            f"Ingesting data from {source} (type: {source_type}, collection: {collection_name})"
        )
        self.performance_monitor.start_timer("data_ingestion")

        documents = self.data_loader.load_and_process(source, source_type)
        self.logger.info(f"Loaded {len(documents)} documents")

        chunker = self._make_chunker(chunk_size, chunk_overlap, chunk_strategy)
        chunks = chunker.chunk_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")

        vs.add_documents(chunks)

        if ret.use_hybrid:
            self.logger.info("Fitting hybrid search on ingested documents")
            ret.fit_hybrid_search(chunks)

        self.performance_monitor.end_timer("data_ingestion")
        self.logger.info(f"Ingestion complete: {len(chunks)} chunks")
        return len(chunks)

    def ingest_document_with_metadata(
        self,
        text: str,
        title: str,
        category: str,
        source: str,
        description: str = "",
        tags: str = "",
        author: str = "",
        collection_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = None,
    ) -> int:
        """
        Ingest a pre-parsed plain-text document with rich metadata.

        Entry point for the GUI upload and the MCP ingest_document tool.

        Args:
            text:            Full plain-text content of the document
            title:           Human-readable title (required)
            category:        Document category (required)
            source:          Source label, e.g. original filename
            description:     Optional freeform description
            tags:            Comma-separated tag string, e.g. "finance,Q4,2026"
            author:          Optional author name
            collection_name: Target collection (defaults to Config.COLLECTION_NAME)

        Returns:
            Number of chunks ingested
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        metadata = DocumentMetadata(
            title=title,
            category=category,
            source=source,
            description=description,
            tags=tags,
            author=author,
        )

        collection_name = collection_name or Config.COLLECTION_NAME
        vs, ret = self._get_or_create_collection(collection_name)

        self.logger.info(f"Ingesting document: '{title}' → collection '{collection_name}'")
        self.performance_monitor.start_timer("data_ingestion")

        document = {"text": text, "metadata": metadata.model_dump()}

        chunker = self._make_chunker(chunk_size, chunk_overlap, chunk_strategy)
        chunks = chunker.chunk_documents([document])
        self.logger.info(f"Created {len(chunks)} chunks from '{title}'")

        vs.add_documents(chunks)

        if ret.use_hybrid:
            ret.fit_hybrid_search(chunks)

        self.performance_monitor.end_timer("data_ingestion")
        self.logger.info(f"Document ingestion complete: {len(chunks)} chunks")
        return len(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = None,
        include_sources: bool = True,
        collection_name: str = None,
        provider: str = None,
        model: str = None,
        api_key: str = None,
        temperature: float = None,
        system_prompt: str = None,
        retrieval_strategy: str = None,
        semantic_weight: float = None,
        conversation_history: list[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question:           User question
            top_k:              Number of context chunks to retrieve
            include_sources:    Whether to include source citations
            collection_name:    Collection to search (defaults to Config.COLLECTION_NAME)
            provider:           LLM provider override ('openai', 'anthropic', 'ollama')
            model:              LLM model override
            api_key:            API key override for the provider
            temperature:        LLM temperature override (0.0–1.0)
            system_prompt:      System prompt override
            retrieval_strategy: Retrieval strategy override (see retriever.RETRIEVAL_STRATEGIES)
            semantic_weight:    Hybrid strategy only — semantic vs keyword balance (0.0–1.0)

        Returns:
            Response dictionary with answer and metadata
        """
        collection_name = collection_name or Config.COLLECTION_NAME
        _, ret = self._get_or_create_collection(collection_name)

        self.logger.info(f"Query in collection '{collection_name}': '{question}'")
        self.performance_monitor.reset()
        self.performance_monitor.start_timer("query_processing")

        # Rewrite follow-up questions into standalone queries before retrieval
        search_question = question
        if conversation_history:
            try:
                search_question = self._rewrite_query(
                    question,
                    conversation_history,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )
            except Exception as exc:
                self.logger.warning(f"Query rewrite failed, using original: {exc}")

        self.performance_monitor.start_timer("retrieval")
        context_data = ret.build_context(
            search_question,
            top_k=top_k,
            strategy=retrieval_strategy,
            semantic_weight=semantic_weight,
        )
        self.performance_monitor.end_timer("retrieval")

        if not context_data["context"]:
            self.logger.warning("No relevant context found")
            return QueryResponse(
                question=question,
                answer="I couldn't find relevant information to answer your question.",
            ).model_dump()

        # Use a per-request Generator when any runtime LLM setting is provided;
        # otherwise fall back to the default Generator initialised at startup.
        if provider or model or api_key or temperature is not None or system_prompt:
            gen = Generator(
                provider=provider,
                model=model,
                api_key=api_key,
                temperature=temperature,
                system_prompt=system_prompt,
                log_level=self.log_level,
            )
        else:
            gen = self.generator

        self.performance_monitor.start_timer("generation")
        if include_sources:
            response = gen.generate_with_sources(
                question, context_data, conversation_history=conversation_history
            )
        else:
            response = gen.generate(
                question,
                context_data["context"],
                conversation_history=conversation_history,
            )
        self.performance_monitor.end_timer("generation")

        response["question"] = question
        response["search_question"] = search_question
        response["context_tokens"] = context_data.get("total_tokens")
        self.performance_monitor.end_timer("query_processing")
        response["performance"] = self.performance_monitor.get_metrics()

        self.logger.info("Query processed successfully")
        return response

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def list_collections(self) -> list[str]:
        """Return all collection names in the vector store."""
        return VectorStore.list_collections()

    def get_collection_stats(self, collection_name: str = None) -> dict[str, Any]:
        """
        Get stats for a specific collection.

        Args:
            collection_name: Collection to inspect (defaults to Config.COLLECTION_NAME)
        """
        collection_name = collection_name or Config.COLLECTION_NAME
        vs, _ = self._get_or_create_collection(collection_name)
        return vs.get_collection_stats()

    def delete_collection(self, collection_name: str) -> None:
        """
        Permanently delete a collection from the vector store.

        Args:
            collection_name: Name of the collection to delete
        """
        self.logger.warning(f"Deleting collection: '{collection_name}'")
        VectorStore.delete_collection(collection_name)
        self._collection_cache.pop(collection_name, None)
        self.logger.info(f"Collection '{collection_name}' deleted")

    # ------------------------------------------------------------------
    # Stats + clear (default collection, for backward compatibility)
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics for the default collection."""
        vector_stats = self.vector_store.get_collection_stats()
        return {
            "vector_store": vector_stats,
            "config": {
                "llm_provider": Config.LLM_PROVIDER,
                "llm_model": Config.LLM_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "top_k": Config.TOP_K_RESULTS,
            },
        }

    def clear_data(self, collection_name: str = None) -> None:
        """
        Clear all documents from a collection and reset its hybrid search index.

        Args:
            collection_name: Collection to clear (defaults to Config.COLLECTION_NAME)
        """
        collection_name = collection_name or Config.COLLECTION_NAME
        vs, ret = self._get_or_create_collection(collection_name)

        self.logger.warning(f"Clearing collection: '{collection_name}'")
        vs.clear_collection()

        if ret.use_hybrid and ret.hybrid_retriever:
            hr = ret.hybrid_retriever
            hr.is_fitted = False
            hr.bm25_retriever = None
            hr.ensemble_retriever = None
            bm25_path = str(Config.BM25_INDEX_PATH)
            if os.path.exists(bm25_path):
                os.remove(bm25_path)

        self.logger.info(f"Collection '{collection_name}' cleared")


def main():
    """Main CLI interface for RAG system."""
    parser = argparse.ArgumentParser(description="RAG Document System")
    parser.add_argument("--ingest", type=str, help="Path to data file to ingest")
    parser.add_argument(
        "--source-type",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Type of data source",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (defaults to COLLECTION_NAME in .env)",
    )
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--list-collections", action="store_true", help="List all collections")
    parser.add_argument("--clear", action="store_true", help="Clear collection data")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    rag = RAGSystem(log_level=args.log_level)

    if args.list_collections:
        collections = rag.list_collections()
        print("Collections:", collections if collections else "(none)")

    if args.clear:
        rag.clear_data(collection_name=args.collection)
        print("✓ Data cleared")

    if args.ingest:
        num_chunks = rag.ingest_data(
            args.ingest, source_type=args.source_type, collection_name=args.collection
        )
        print(f"✓ Ingested data: {num_chunks} chunks added")

    if args.stats:
        stats = rag.get_stats()
        print("\n=== System Statistics ===")
        print(f"Documents: {stats['vector_store']['document_count']}")
        print(f"LLM: {stats['config']['llm_provider']} / {stats['config']['llm_model']}")
        print(f"Embedding: {stats['config']['embedding_model']}")
        print(f"Chunk Size: {stats['config']['chunk_size']}")

    if args.query:
        response = rag.query(args.query, top_k=args.top_k, collection_name=args.collection)
        print(f"\n=== Question ===\n{response['question']}")
        print(f"\n=== Answer ===\n{response.get('answer_with_citations', response['answer'])}")
        print("\n=== Metadata ===")
        print(f"Sources: {response.get('num_sources', 0)}")
        print(f"Model: {response.get('model', 'N/A')}")
        print(f"Tokens: {response.get('tokens_used', 'N/A')}")

    if args.interactive:
        collection = args.collection or Config.COLLECTION_NAME
        print(f"\n=== Interactive RAG System (collection: {collection}) ===")
        print("Commands: 'exit', 'stats', 'collections'\n")

        while True:
            try:
                question = input("Question: ").strip()

                if question.lower() == "exit":
                    break
                elif question.lower() == "stats":
                    stats = rag.get_collection_stats(collection)
                    print(f"Documents: {stats['document_count']}")
                    continue
                elif question.lower() == "collections":
                    print("Collections:", rag.list_collections())
                    continue
                elif not question:
                    continue

                response = rag.query(question, top_k=args.top_k, collection_name=collection)
                print(f"\nAnswer: {response.get('answer_with_citations', response['answer'])}\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
