"""
Test hybrid search functionality using LangChainHybridRetriever.
"""

import pytest
from hybrid_search import LangChainHybridRetriever


SAMPLE_DOCS = [
    {"text": "Q4 revenue increased by 25%", "metadata": {"category": "finance"}},
    {"text": "Daily active users reached 1 million", "metadata": {"category": "analytics"}},
    {"text": "Customer satisfaction score is 4.5", "metadata": {"category": "feedback"}},
]


class TestLangChainHybridRetriever:
    """Test LangChainHybridRetriever."""

    def test_initialization(self):
        """Weights are normalized to sum to 1."""
        hybrid = LangChainHybridRetriever(semantic_weight=0.6, keyword_weight=0.4)
        assert abs(hybrid.semantic_weight + hybrid.keyword_weight - 1.0) < 1e-6

    def test_initialization_unbalanced_weights(self):
        """Non-unit weights are normalized correctly."""
        hybrid = LangChainHybridRetriever(semantic_weight=7, keyword_weight=3)
        assert abs(hybrid.semantic_weight - 0.7) < 1e-6
        assert abs(hybrid.keyword_weight - 0.3) < 1e-6

    def test_not_fitted_on_init(self):
        """Retriever starts unfitted."""
        hybrid = LangChainHybridRetriever()
        assert not hybrid.is_fitted

    def test_fit(self):
        """Fitting marks retriever as ready."""
        hybrid = LangChainHybridRetriever()
        hybrid.fit(SAMPLE_DOCS)
        assert hybrid.is_fitted
        assert hybrid.bm25_retriever is not None

    def test_fit_empty_documents(self):
        """Fitting on empty list leaves retriever unfitted."""
        hybrid = LangChainHybridRetriever()
        hybrid.fit([])
        assert not hybrid.is_fitted

    def test_search_returns_empty_when_not_fitted(self):
        """Search before fit returns empty list."""
        hybrid = LangChainHybridRetriever()
        results = hybrid.search("revenue", vector_retriever=None, top_k=3)
        assert results == []

    def test_search_with_vector_retriever(self):
        """Hybrid search returns results with correct keys."""
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        lc_docs = [Document(page_content=d["text"], metadata=d["metadata"]) for d in SAMPLE_DOCS]
        chroma = Chroma.from_documents(lc_docs, embeddings)
        vector_retriever = chroma.as_retriever(search_kwargs={"k": 10})

        hybrid = LangChainHybridRetriever(semantic_weight=0.7, keyword_weight=0.3)
        hybrid.fit(SAMPLE_DOCS)

        results = hybrid.search("Q4 revenue", vector_retriever, top_k=2)

        assert len(results) <= 2
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "rank" in r

        # Cleanup
        chroma.delete_collection()

    def test_save_and_load_index(self, tmp_path):
        """BM25 index can be saved and loaded."""
        hybrid = LangChainHybridRetriever()
        hybrid.fit(SAMPLE_DOCS)

        index_path = str(tmp_path / "bm25.pkl")
        hybrid.save_index(index_path)

        # Load into a fresh instance
        hybrid2 = LangChainHybridRetriever()
        assert not hybrid2.is_fitted
        hybrid2.load_index(index_path)
        assert hybrid2.is_fitted

    def test_load_nonexistent_index(self, tmp_path):
        """Loading a missing index path logs a warning but doesn't raise."""
        hybrid = LangChainHybridRetriever()
        hybrid.load_index(str(tmp_path / "missing.pkl"))  # should not raise
        assert not hybrid.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
