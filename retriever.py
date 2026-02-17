"""
Retrieval module for RAG system.
Handles query processing and context retrieval from vector store.
Supports hybrid search combining semantic and keyword-based retrieval.
"""

import logging
from typing import List, Dict, Any, Optional

from config import Config
from vector_store import VectorStore
from hybrid_search import LangChainHybridRetriever, LangChainVectorRetrieverWrapper
from utils import setup_logging, timer, format_context_for_llm, count_tokens


class Retriever:
    """Retrieve relevant context for queries using hybrid search."""

    def __init__(self, vector_store: VectorStore = None, log_level: str = "INFO"):
        """
        Initialize Retriever.

        Args:
            vector_store: VectorStore instance (creates new if None)
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.vector_store = vector_store or VectorStore(log_level=log_level)
        self.top_k = Config.TOP_K_RESULTS
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        self.max_context_length = Config.MAX_CONTEXT_LENGTH

        # Initialize LangChain hybrid search
        self.use_hybrid = Config.USE_HYBRID_SEARCH
        if self.use_hybrid:
            self.hybrid_retriever = LangChainHybridRetriever(
                semantic_weight=Config.SEMANTIC_WEIGHT,
                keyword_weight=Config.KEYWORD_WEIGHT,
                log_level=log_level,
            )
            # Wrap our vector store for LangChain compatibility
            self.vector_retriever_wrapper = LangChainVectorRetrieverWrapper(
                vector_store=self.vector_store,
                k=20,  # Retrieve more for ensemble fusion
            )
            self.logger.info(
                f"LangChain hybrid search enabled "
                f"(semantic: {Config.SEMANTIC_WEIGHT:.0%}, keyword: {Config.KEYWORD_WEIGHT:.0%})"
            )
        else:
            self.hybrid_retriever = None
            self.vector_retriever_wrapper = None
            self.logger.info("Using semantic search only")

        # Attempt to load saved hybrid index on initialization
        if self.use_hybrid and self.hybrid_retriever:
            self.load_hybrid_index()

    def fit_hybrid_search(self, documents: List[Dict[str, Any]]):
        """
        Fit the hybrid search BM25 component on documents.
        Should be called after adding documents to vector store.

        Args:
            documents: List of documents with 'text' field
        """
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.fit(documents)
            self.save_hybrid_index()
            self.logger.info("Hybrid search fitted on documents")

    def save_hybrid_index(self):
        """Save the hybrid search index."""
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.save_index(Config.BM25_INDEX_PATH)

    def load_hybrid_index(self):
        """Load the hybrid search index."""
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.load_index(Config.BM25_INDEX_PATH)

    @timer
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query
            top_k: Number of results to retrieve
            metadata_filter: Optional metadata filter
            use_hybrid: Override hybrid search setting (None uses config default)

        Returns:
            List of relevant documents
        """
        top_k = top_k or self.top_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid

        self.logger.info(
            f"Retrieving context for query: '{query}' (hybrid={use_hybrid})"
        )

        # Use LangChain hybrid search if enabled and fitted

        if use_hybrid and self.hybrid_retriever and self.hybrid_retriever.is_fitted:
            self.logger.debug("Using LangChain EnsembleRetriever")
            results = self.hybrid_retriever.search(
                query, self.vector_retriever_wrapper, top_k=top_k
            )

            # Add similarity scores (not provided by ensemble, use rank-based)
            for i, result in enumerate(results):
                # Inverse rank as similarity score
                result["similarity_score"] = 1.0 / (i + 1)

            final_results = results
        else:
            # Fall back to semantic search only
            self.logger.debug("Using semantic search only")
            semantic_results = self.vector_store.search(
                query=query, top_k=top_k, metadata_filter=metadata_filter
            )

            # Filter by similarity threshold and add similarity scores
            final_results = []
            for result in semantic_results:
                distance = result.get("distance", 0)
                # Convert distance to similarity score (inverse relationship)
                similarity = 1 / (1 + distance)
                result["similarity_score"] = similarity

                if similarity >= self.similarity_threshold:
                    final_results.append(result)

        self.logger.info(f"Retrieved {len(final_results)} relevant documents")
        return final_results

    def retrieve_with_reranking(
        self, query: str, top_k: int = None, initial_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with two-stage retrieval (retrieve more, then rerank).

        Args:
            query: User query
            top_k: Final number of results to return
            initial_k: Number of results to retrieve initially (before reranking)

        Returns:
            List of reranked relevant documents
        """
        top_k = top_k or self.top_k
        initial_k = initial_k or (top_k * 3)  # Retrieve 3x more initially

        self.logger.info(
            f"Retrieving with reranking (initial_k={initial_k}, final_k={top_k})"
        )

        # Initial retrieval
        initial_results = self.retrieve(query, top_k=initial_k)

        # Simple reranking: sort by similarity score
        # In production, you could use a cross-encoder model here
        reranked_results = sorted(
            initial_results, key=lambda x: x.get("similarity_score", 0), reverse=True
        )[:top_k]

        self.logger.info(f"Reranked to {len(reranked_results)} documents")
        return reranked_results

    def build_context(
        self, query: str, top_k: int = None, use_reranking: bool = None
    ) -> Dict[str, Any]:
        """
        Build context for LLM generation.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_reranking: Whether to use reranking

        Returns:
            Dictionary with context string and source documents
        """
        use_reranking = (
            use_reranking if use_reranking is not None else Config.USE_RERANKING
        )

        # Retrieve documents
        if use_reranking:
            documents = self.retrieve_with_reranking(query, top_k=top_k)
        else:
            documents = self.retrieve(query, top_k=top_k)

        if not documents:
            self.logger.warning("No relevant documents found")
            return {"context": "", "sources": [], "num_sources": 0}

        # Format context for LLM
        context = format_context_for_llm(documents)

        # Truncate if too long
        context_tokens = count_tokens(context)
        if context_tokens > self.max_context_length:
            self.logger.warning(
                f"Context too long ({context_tokens} tokens), truncating to {self.max_context_length}"
            )
            # Remove documents from the end until within limit
            while context_tokens > self.max_context_length and len(documents) > 1:
                documents.pop()
                context = format_context_for_llm(documents)
                context_tokens = count_tokens(context)

        return {
            "context": context,
            "sources": documents,
            "num_sources": len(documents),
            "total_tokens": context_tokens,
        }

    def retrieve_by_metadata(
        self, metadata_filter: Dict[str, Any], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents by metadata filter only.

        Args:
            metadata_filter: Metadata filter criteria
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        top_k = top_k or self.top_k

        self.logger.info(f"Retrieving by metadata: {metadata_filter}")

        # Use a generic query since we're filtering by metadata
        results = self.vector_store.search(
            query="", top_k=top_k, metadata_filter=metadata_filter
        )

        return results


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from chunking import TextChunker

    # Initialize components
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store)

    # Sample data
    sample_data = [
        {
            "title": "Q4 Revenue Report",
            "content": "Revenue increased by 25% in Q4 2025. Total revenue reached $10M.",
            "category": "finance",
            "date": "2025-12-31",
        },
        {
            "title": "User Engagement Metrics",
            "content": "Daily active users reached 1M milestone. User retention improved by 15%.",
            "category": "analytics",
            "date": "2026-01-15",
        },
        {
            "title": "Customer Satisfaction",
            "content": "Customer satisfaction score improved to 4.5 out of 5. NPS increased to 45.",
            "category": "feedback",
            "date": "2026-01-20",
        },
    ]

    # Load and process data
    loader = DataLoader()
    documents = loader.process_records(sample_data)

    # Chunk documents
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.chunk_documents(documents)

    # Add to vector store
    vector_store.add_documents(chunks)

    # Test retrieval
    print("\n=== Testing Retrieval ===")
    query = "How is our revenue performing?"
    context_data = retriever.build_context(query, top_k=2)

    print(f"\nQuery: {query}")
    print(f"Found {context_data['num_sources']} sources")
    print(f"Context tokens: {context_data['total_tokens']}")
    print(f"\nContext:\n{context_data['context']}")
