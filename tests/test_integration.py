"""
Integration tests for the complete RAG system.
"""

import os

import chromadb.errors
import pytest

from raglab.config import Config
from raglab.rag_system import RAGSystem
from raglab.retrieval.retriever import bm25_index_path
from raglab.retrieval.vector_store import VectorStore

TEST_COLLECTION = "test-integration"


class TestRAGSystemIntegration:
    """Integration tests for end-to-end RAG functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, monkeypatch):
        """Set up and tear down test fixtures, isolated from the real default collection."""
        # Isolate: point Config.COLLECTION_NAME at a dedicated test collection so this
        # suite never touches the real default collection used by the GUI/CLI/MCP server.
        monkeypatch.setattr(Config, "COLLECTION_NAME", TEST_COLLECTION)

        # Guard against a prior interrupted run leaving stale data behind.
        try:
            VectorStore.delete_collection(TEST_COLLECTION)
        except chromadb.errors.NotFoundError:
            pass
        test_bm25_path = bm25_index_path(TEST_COLLECTION)
        if test_bm25_path.exists():
            test_bm25_path.unlink()

        # Setup
        self.rag = RAGSystem(log_level="WARNING")  # Reduce log noise in tests

        # Sample test data
        self.test_data = [
            {
                "id": 1,
                "title": "Revenue Report",
                "content": "Q4 revenue was $10M, up 25% from last year",
                "category": "finance",
            },
            {
                "id": 2,
                "title": "User Metrics",
                "content": "We have 1 million daily active users",
                "category": "analytics",
            },
        ]

        yield

        # Teardown: clear_data() also resets this collection's BM25 index file,
        # which delete_collection() alone would leave behind.
        self.rag.clear_data(TEST_COLLECTION)
        self.rag.delete_collection(TEST_COLLECTION)

    def test_data_ingestion(self):
        """Test complete data ingestion pipeline."""
        # Ingest from dict
        num_chunks = self.rag.ingest_data(source=self.test_data, source_type="dict")

        assert num_chunks > 0, "Should create chunks from data"

        # Verify data is in vector store
        stats = self.rag.get_stats()
        assert stats["vector_store"]["document_count"] > 0, "Should have documents in store"

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
        reason="No LLM API key available",
    )
    def test_end_to_end_query(self):
        """Test complete query pipeline (requires API key)."""
        # Ingest data
        self.rag.ingest_data(source=self.test_data, source_type="dict")

        # Query
        response = self.rag.query("What is our revenue?", top_k=2)

        # Validate response structure
        assert "question" in response, "Should have question"
        assert "answer" in response, "Should have answer"
        assert "sources" in response, "Should have sources"
        assert len(response["answer"]) > 0, "Answer should not be empty"

    def test_query_without_data(self):
        """Test querying empty system."""
        # Don't ingest any data
        response = self.rag.query("What is our revenue?", top_k=2)

        # Should handle gracefully
        assert "answer" in response, "Should have answer field"
        assert response["num_sources"] == 0, "Should have no sources"

    def test_system_stats(self):
        """Test system statistics."""
        stats = self.rag.get_stats()

        assert "vector_store" in stats, "Should have vector store stats"
        assert "config" in stats, "Should have config stats"
        assert "document_count" in stats["vector_store"], "Should have document count"

    def test_clear_data(self):
        """Test data clearing."""
        # Ingest data
        self.rag.ingest_data(source=self.test_data, source_type="dict")

        # Verify data exists
        stats_before = self.rag.get_stats()
        assert stats_before["vector_store"]["document_count"] > 0

        # Clear
        self.rag.clear_data()

        # Verify data is gone
        stats_after = self.rag.get_stats()
        assert stats_after["vector_store"]["document_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
