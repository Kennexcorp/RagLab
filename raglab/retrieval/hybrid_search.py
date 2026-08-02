"""
Hybrid search module using LangChain's EnsembleRetriever.
Combines semantic vector search with BM25 keyword search.
"""

import pickle
from pathlib import Path
from typing import Any

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from raglab.utils import setup_logging


class LangChainHybridRetriever:
    """
    Hybrid retriever using LangChain's EnsembleRetriever.
    Combines BM25 keyword search with semantic vector search.

    The vector_retriever passed to search() and create_ensemble() must be a
    LangChain BaseRetriever — use VectorStore.as_retriever() to obtain one.
    """

    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        log_level: str = "INFO",
    ):
        """
        Initialize LangChain hybrid retriever.

        Args:
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        # Normalize weights
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total

        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.is_fitted = False

        self.logger.info(
            f"LangChain hybrid retriever initialized "
            f"(semantic: {self.semantic_weight:.0%}, keyword: {self.keyword_weight:.0%})"
        )

    def fit(self, documents: list[dict[str, Any]]):
        """
        Fit the BM25 retriever on documents.

        Args:
            documents: List of documents with 'text' and 'metadata' fields
        """
        if not documents:
            self.logger.warning("No documents to fit")
            return

        lc_docs = [
            Document(page_content=doc["text"], metadata=doc.get("metadata", {}))
            for doc in documents
        ]

        self.bm25_retriever = BM25Retriever.from_documents(lc_docs)
        self.bm25_retriever.k = 20  # Retrieve more for fusion
        self.ensemble_retriever = None  # Reset so it's rebuilt on next search
        self.is_fitted = True
        self.logger.info(f"BM25 retriever fitted on {len(documents)} documents")

    def save_index(self, path: str):
        """
        Save the fitted BM25 retriever to disk.

        Args:
            path: Path to save result
        """
        if not self.is_fitted or not self.bm25_retriever:
            self.logger.warning("Cannot save: BM25 retriever not fitted")
            return

        try:
            with open(path, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            self.logger.info(f"Saved BM25 index to {path}")
        except Exception as e:
            self.logger.error(f"Failed to save BM25 index: {str(e)}")

    def load_index(self, path: str):
        """
        Load the fitted BM25 retriever from disk.

        Args:
            path: Path to load from
        """
        path = Path(path)
        if not path.exists():
            self.logger.warning(f"No BM25 index found at {path}")
            return

        try:
            with open(path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            self.is_fitted = True
            self.ensemble_retriever = None
            self.logger.info(f"Loaded BM25 index from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {str(e)}")

    def create_ensemble(self, vector_retriever):
        """
        Create ensemble retriever combining BM25 and vector search.

        Args:
            vector_retriever: LangChain BaseRetriever (e.g. VectorStore.as_retriever())
        """
        if not self.is_fitted:
            self.logger.error("BM25 retriever not fitted. Call fit() first.")
            return None

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[self.keyword_weight, self.semantic_weight],
        )

        self.logger.info("Ensemble retriever created")
        return self.ensemble_retriever

    def search(self, query: str, vector_retriever, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Perform hybrid search using ensemble retriever.

        Args:
            query: Search query
            vector_retriever: LangChain BaseRetriever (from VectorStore.as_retriever())
            top_k: Number of results to return

        Returns:
            List of search results with text, metadata, and rank
        """
        if not self.is_fitted:
            self.logger.warning("BM25 not fitted, cannot perform hybrid search")
            return []

        if self.ensemble_retriever is None:
            self.create_ensemble(vector_retriever)

        results = self.ensemble_retriever.invoke(query)

        formatted = [
            {"text": doc.page_content, "metadata": doc.metadata, "rank": i + 1}
            for i, doc in enumerate(results[:top_k])
        ]

        self.logger.info(f"Hybrid search returned {len(formatted)} results")
        return formatted


if __name__ == "__main__":
    # Example usage
    from raglab.retrieval.vector_store import VectorStore

    sample_docs = [
        {
            "text": "Q4 revenue increased by 25% to $10M",
            "metadata": {"category": "finance", "quarter": "Q4"},
        },
        {
            "text": "Daily active users reached 1 million",
            "metadata": {"category": "analytics", "metric": "DAU"},
        },
        {
            "text": "Customer satisfaction score is 4.5 out of 5",
            "metadata": {"category": "feedback", "metric": "CSAT"},
        },
    ]

    vector_store = VectorStore()
    vector_store.add_documents(sample_docs)

    hybrid = LangChainHybridRetriever(semantic_weight=0.7, keyword_weight=0.3)
    hybrid.fit(sample_docs)

    query = "Q4 revenue performance"
    results = hybrid.search(query, vector_store.as_retriever(k=10), top_k=3)

    print(f"\nHybrid search results for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Metadata: {result['metadata']}")
