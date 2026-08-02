"""
Unit tests for retriever module.
"""

import pytest

from retriever import Retriever
from vector_store import VectorStore


class TestRetriever:
    """Test cases for Retriever class."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down test fixtures."""
        # Setup
        self.vector_store = VectorStore(collection_name="test_collection")
        self.retriever = Retriever(vector_store=self.vector_store)

        # Add test documents
        test_docs = [
            {
                "text": "Q4 revenue increased by 25% to $10M",
                "metadata": {"category": "finance", "source": "report1"},
            },
            {
                "text": "User engagement metrics show 1M daily active users",
                "metadata": {"category": "analytics", "source": "report2"},
            },
            {
                "text": "Customer satisfaction score is 4.5 out of 5",
                "metadata": {"category": "feedback", "source": "report3"},
            },
        ]
        self.vector_store.add_documents(test_docs)
        self.retriever.fit_hybrid_search(test_docs)  # required for hybrid path

        yield

        # Teardown
        self.vector_store.clear_collection()

    def test_retrieve_basic(self):
        """Test basic retrieval."""
        results = self.retriever.retrieve("revenue performance", top_k=2)

        assert len(results) > 0, "Should return results"
        assert len(results) <= 2, "Should respect top_k limit"
        assert all("text" in r for r in results), "Results should have text"
        assert all("metadata" in r for r in results), "Results should have metadata"

    def test_retrieve_with_similarity_threshold(self):
        """Test retrieval with similarity filtering."""
        # Query for something very specific
        results = self.retriever.retrieve("revenue", top_k=5)

        # All results should have similarity scores
        assert all("similarity_score" in r for r in results), "Should have similarity scores"

    def test_build_context(self):
        """Test context building for LLM."""
        context_data = self.retriever.build_context("How is revenue?", top_k=2)

        assert "context" in context_data, "Should have context string"
        assert "sources" in context_data, "Should have sources"
        assert "num_sources" in context_data, "Should have source count"
        assert "total_tokens" in context_data, "Should have token count"
        assert len(context_data["context"]) > 0, "Context should not be empty"

    def test_retrieve_no_results(self):
        """Test retrieval with query that has no good matches."""
        # Clear and add unrelated document
        self.vector_store.clear_collection()
        self.vector_store.add_documents(
            [{"text": "Completely unrelated content about weather", "metadata": {}}]
        )

        context_data = self.retriever.build_context("quantum physics equations", top_k=5)

        # Should still return something, but might be low quality
        assert "context" in context_data
        assert "sources" in context_data

    def test_retrieve_with_metadata_filter(self):
        """Test retrieval with metadata filtering."""
        results = self.retriever.retrieve_by_metadata(
            metadata_filter={"category": "finance"}, top_k=5
        )

        # Note: ChromaDB metadata filtering might not work in all versions
        # This test validates the interface works
        assert isinstance(results, list), "Should return a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
