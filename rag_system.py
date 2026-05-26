"""
Main RAG System orchestrator.
Coordinates all components to provide end-to-end question answering.
"""

import logging
import argparse
import os
from typing import Dict, Any, Optional, List

from config import Config
from data_loader import DataLoader
from chunking import TextChunker
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator
from utils import setup_logging, PerformanceMonitor


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
        self.logger.info("Initializing RAG System")

        # Validate configuration
        if not Config.validate():
            raise ValueError("Configuration validation failed")

        # Initialize components
        self.data_loader = DataLoader(log_level=log_level)
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            strategy=Config.CHUNKING_STRATEGY,
            log_level=log_level,
        )
        self.vector_store = VectorStore(log_level=log_level)
        self.retriever = Retriever(vector_store=self.vector_store, log_level=log_level)
        self.generator = Generator(log_level=log_level)

        self.performance_monitor = PerformanceMonitor()

        self.logger.info("RAG System initialized successfully")
        self.logger.info(Config.display_config())

    def ingest_data(self, source: str, source_type: str = "json") -> int:
        """
        Ingest data from a source into the RAG system.

        Args:
            source: Path to data file or data dict
            source_type: Type of source ('json', 'csv', 'dict')

        Returns:
            Number of chunks ingested
        """
        self.logger.info(f"Ingesting data from {source} (type: {source_type})")
        self.performance_monitor.start_timer("data_ingestion")

        # Load data
        documents = self.data_loader.load_and_process(source, source_type)
        self.logger.info(f"Loaded {len(documents)} documents")

        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        self.logger.info(f"Created {len(chunks)} chunks")

        # Add to vector store
        self.vector_store.add_documents(chunks)

        # Fit hybrid search BM25 component
        if self.retriever.use_hybrid:
            self.logger.info("Fitting hybrid search on ingested documents")
            self.retriever.fit_hybrid_search(chunks)

        self.performance_monitor.end_timer("data_ingestion")
        self.logger.info(f"Ingestion complete: {len(chunks)} chunks")

        return len(chunks)

    def ingest_keytable(self, file_path: str) -> int:
        """
        Ingest a keytable JSON file into the RAG system.

        Each row in the series tree becomes one document — chunking is skipped
        because rows are already properly sized for embedding.  Multiple calls
        with different dates accumulate in the same collection, enabling
        cross-date semantic search and per-date metadata filtering.

        Args:
            file_path: Path to the keytable JSON file

        Returns:
            Number of documents ingested
        """
        from keytable_loader import KeytableLoader

        self.logger.info(f"Ingesting keytable from {file_path}")
        self.performance_monitor.start_timer("data_ingestion")

        loader = KeytableLoader(log_level="INFO")
        documents = loader.load(file_path)

        if not documents:
            self.logger.warning("No documents extracted from keytable")
            return 0

        self.vector_store.add_documents(documents)

        if self.retriever.use_hybrid:
            self.retriever.fit_hybrid_search(documents)

        self.performance_monitor.end_timer("data_ingestion")
        self.logger.info(f"Keytable ingestion complete: {len(documents)} documents")
        return len(documents)

    def query(
        self, question: str, top_k: int = None, include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            include_sources: Whether to include source citations

        Returns:
            Response dictionary with answer and metadata
        """
        self.logger.info(f"Processing query: '{question}'")
        self.performance_monitor.reset()
        self.performance_monitor.start_timer("query_processing")

        # Retrieve context
        self.performance_monitor.start_timer("retrieval")
        context_data = self.retriever.build_context(question, top_k=top_k)
        self.performance_monitor.end_timer("retrieval")

        if not context_data["context"]:
            self.logger.warning("No relevant context found")
            return {
                "question": question,
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "num_sources": 0,
            }

        # Generate answer
        self.performance_monitor.start_timer("generation")
        if include_sources:
            response = self.generator.generate_with_sources(question, context_data)
        else:
            response = self.generator.generate(question, context_data["context"])
        self.performance_monitor.end_timer("generation")

        # Add question to response
        response["question"] = question

        self.performance_monitor.end_timer("query_processing")

        # Add performance metrics
        response["performance"] = self.performance_monitor.get_metrics()

        self.logger.info("Query processed successfully")
        return response

    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with system stats
        """
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

    def clear_data(self):
        """Clear all data from the vector store and reset hybrid search index."""
        self.logger.warning("Clearing all data from vector store")
        self.vector_store.clear_collection()

        # Reset the hybrid search BM25 state so stale index data doesn't persist
        if self.retriever.use_hybrid and self.retriever.hybrid_retriever:
            hr = self.retriever.hybrid_retriever
            hr.is_fitted = False
            hr.bm25_retriever = None
            hr.ensemble_retriever = None
            if os.path.exists(Config.BM25_INDEX_PATH):
                os.remove(Config.BM25_INDEX_PATH)

        self.logger.info("Data cleared successfully")


def main():
    """Main CLI interface for RAG system."""
    parser = argparse.ArgumentParser(
        description="RAG System for Organization Dashboard"
    )
    parser.add_argument("--ingest", type=str, help="Path to data file to ingest")
    parser.add_argument(
        "--source-type",
        type=str,
        default="json",
        choices=["json", "csv"],
        help="Type of data source",
    )
    parser.add_argument(
        "--ingest-keytable",
        type=str,
        help="Path to keytable JSON file to ingest",
    )
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to retrieve"
    )
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--clear", action="store_true", help="Clear all data")
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive mode"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag = RAGSystem(log_level=args.log_level)

    # Handle commands
    if args.clear:
        rag.clear_data()
        print("✓ Data cleared")

    if args.ingest:
        num_chunks = rag.ingest_data(args.ingest, source_type=args.source_type)
        print(f"✓ Ingested data: {num_chunks} chunks added")

    if args.ingest_keytable:
        num_docs = rag.ingest_keytable(args.ingest_keytable)
        print(f"✓ Keytable ingested: {num_docs} documents added")

    if args.stats:
        stats = rag.get_stats()
        print("\n=== System Statistics ===")
        print(f"Documents: {stats['vector_store']['document_count']}")
        print(
            f"LLM: {stats['config']['llm_provider']} / {stats['config']['llm_model']}"
        )
        print(f"Embedding: {stats['config']['embedding_model']}")
        print(f"Chunk Size: {stats['config']['chunk_size']}")

    if args.query:
        response = rag.query(args.query, top_k=args.top_k)
        print(f"\n=== Question ===\n{response['question']}")
        print(
            f"\n=== Answer ===\n{response.get('answer_with_citations', response['answer'])}"
        )
        print(f"\n=== Metadata ===")
        print(f"Sources: {response.get('num_sources', 0)}")
        print(f"Model: {response.get('model', 'N/A')}")
        print(f"Tokens: {response.get('tokens_used', 'N/A')}")

    if args.interactive:
        print("\n=== Interactive RAG System ===")
        print("Commands: 'exit', 'stats', 'ingest-keytable <path>'\n")

        while True:
            try:
                question = input("Question: ").strip()

                if question.lower() == "exit":
                    break
                elif question.lower() == "stats":
                    stats = rag.get_stats()
                    print(f"Documents: {stats['vector_store']['document_count']}")
                    continue
                elif question.lower().startswith("ingest-keytable "):
                    path = question[len("ingest-keytable "):].strip()
                    num_docs = rag.ingest_keytable(path)
                    print(f"✓ Keytable ingested: {num_docs} documents added\n")
                    continue
                elif not question:
                    continue

                response = rag.query(question, top_k=args.top_k)
                print(
                    f"\nAnswer: {response.get('answer_with_citations', response['answer'])}\n"
                )

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
