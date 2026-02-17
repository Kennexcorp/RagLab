"""
Test hybrid search functionality.
"""

import pytest
from hybrid_search import BM25, HybridSearcher


class TestBM25:
    """Test BM25 keyword search."""

    def test_bm25_basic_search(self):
        """Test basic BM25 search."""
        documents = [
            "Q4 revenue increased by 25%",
            "Daily active users reached 1 million",
            "Customer satisfaction score is 4.5",
        ]

        bm25 = BM25()
        bm25.fit(documents)

        results = bm25.search("revenue Q4", top_k=2)

        assert len(results) > 0
        assert results[0]["index"] == 0  # First doc should match best
        assert results[0]["score"] > 0

    def test_bm25_no_match(self):
        """Test BM25 with no matching terms."""
        documents = [
            "Q4 revenue increased by 25%",
            "Daily active users reached 1 million",
        ]

        bm25 = BM25()
        bm25.fit(documents)

        results = bm25.search("xyz abc def", top_k=5)

        # Should return empty or zero scores
        assert len(results) == 0 or all(r["score"] == 0 for r in results)

    def test_bm25_parameters(self):
        """Test BM25 with custom parameters."""
        documents = ["test document one", "test document two"]

        bm25 = BM25(k1=2.0, b=0.5)
        bm25.fit(documents)

        results = bm25.search("test", top_k=2)
        assert len(results) == 2


class TestHybridSearcher:
    """Test hybrid search combining semantic and keyword."""

    def test_hybrid_initialization(self):
        """Test hybrid searcher initialization."""
        hybrid = HybridSearcher(semantic_weight=0.6, keyword_weight=0.4)

        # Weights should be normalized
        assert abs(hybrid.semantic_weight + hybrid.keyword_weight - 1.0) < 0.01

    def test_hybrid_fit(self):
        """Test fitting hybrid search on documents."""
        documents = [
            {"text": "Q4 revenue increased by 25%"},
            {"text": "Daily active users reached 1 million"},
        ]

        hybrid = HybridSearcher()
        hybrid.fit(documents)

        assert hybrid.is_fitted
        assert len(hybrid.documents) == 2

    def test_hybrid_search(self):
        """Test hybrid search with semantic and keyword results."""
        documents = [
            {"text": "Q4 revenue increased by 25%"},
            {"text": "Daily active users reached 1 million"},
            {"text": "Customer satisfaction improved"},
        ]

        hybrid = HybridSearcher(semantic_weight=0.7, keyword_weight=0.3)
        hybrid.fit(documents)

        # Simulate semantic results
        semantic_results = [
            {"text": documents[0]["text"], "similarity_score": 0.9},
            {"text": documents[1]["text"], "similarity_score": 0.5},
            {"text": documents[2]["text"], "similarity_score": 0.3},
        ]

        results = hybrid.search("Q4 revenue", semantic_results, top_k=2)

        assert len(results) <= 2
        assert "hybrid_score" in results[0]
        assert "semantic_score" in results[0]
        assert "keyword_score" in results[0]

    def test_score_normalization(self):
        """Test score normalization."""
        hybrid = HybridSearcher()

        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = hybrid._normalize_scores(scores)

        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
        assert len(normalized) == len(scores)

    def test_score_normalization_edge_cases(self):
        """Test score normalization edge cases."""
        hybrid = HybridSearcher()

        # All zeros
        assert hybrid._normalize_scores([0, 0, 0]) == [0, 0, 0]

        # All same values
        assert hybrid._normalize_scores([5, 5, 5]) == [1.0, 1.0, 1.0]

        # Empty list
        assert hybrid._normalize_scores([]) == []

    def test_hybrid_without_fit(self):
        """Test hybrid search without fitting (should fallback to semantic)."""
        hybrid = HybridSearcher()

        semantic_results = [{"text": "test document", "similarity_score": 0.8}]

        results = hybrid.search("test", semantic_results, top_k=1)

        # Should return semantic results only
        assert len(results) == 1
        assert results[0]["text"] == "test document"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
