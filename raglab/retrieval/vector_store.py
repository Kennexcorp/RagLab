"""
Vector store module for RAG system.
Uses LangChain's Chroma integration and HuggingFaceEmbeddings for simplified
embedding generation and vector database operations.
"""

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from raglab.config import Config
from raglab.utils import setup_logging, timer


class VectorStore:
    """Manage vector embeddings and similarity search using LangChain's Chroma."""

    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        persist_directory: str = None,
        log_level: str = "INFO",
    ):
        """
        Initialize VectorStore.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
            persist_directory: Directory to persist the database
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.persist_directory = persist_directory or Config.PERSIST_DIRECTORY

        self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        self.logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        self.logger.info(f"Vector store initialized with collection: {self.collection_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logging.getLogger("RAG_System"), logging.WARNING),
        reraise=True,
    )
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents with 'text' and 'metadata' fields
        """
        if not documents:
            self.logger.warning("No documents to add")
            return

        self.logger.info(f"Adding {len(documents)} documents to vector store")

        lc_docs = []
        for doc in documents:
            # Stringify metadata values — ChromaDB requirement
            metadata = {k: str(v) for k, v in doc.get("metadata", {}).items() if v is not None}
            lc_docs.append(Document(page_content=doc["text"], metadata=metadata))

        self.chroma.add_documents(lc_docs)
        self.logger.info(f"Successfully added {len(documents)} documents")

    @timer
    def search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of search results with text, metadata, and distance
        """
        top_k = top_k or Config.TOP_K_RESULTS
        self.logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Use similarity_search_with_score which returns raw L2 distances
        # (lower = more similar), matching the behaviour of the old chromadb client.
        results = self.chroma.similarity_search_with_score(query, k=top_k, filter=metadata_filter)

        formatted = [
            {"text": doc.page_content, "metadata": doc.metadata, "distance": score}
            for doc, score in results
        ]

        self.logger.info(f"Found {len(formatted)} results")
        return formatted

    def as_retriever(self, k: int = 20, search_type: str = "similarity"):
        """
        Return a LangChain BaseRetriever for use in chains and EnsembleRetriever.

        Args:
            k: Number of documents to retrieve
            search_type: 'similarity' (default) or 'mmr' for diverse results
        """
        return self.chroma.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )

    def get_by_metadata(
        self, metadata_filter: dict[str, Any], top_k: int = None
    ) -> list[dict[str, Any]]:
        """
        Fetch documents by metadata filter without a similarity search.

        Uses Chroma's .get() API so no query embedding is needed — results
        are deterministic and not ranked by relevance.

        Args:
            metadata_filter: ChromaDB where-clause dict, e.g. {"category": "finance"}
            top_k: Maximum number of results (None = all matches)

        Returns:
            List of matching documents with text and metadata
        """
        self.logger.info(f"Fetching by metadata: {metadata_filter}")
        kwargs = {"where": metadata_filter}
        if top_k:
            kwargs["limit"] = top_k

        result = self.chroma._collection.get(**kwargs)

        docs = [
            {"text": text, "metadata": meta}
            for text, meta in zip(result["documents"], result["metadatas"], strict=True)
        ]
        self.logger.info(f"Found {len(docs)} documents")
        return docs

    def get_all_documents(self) -> list[dict[str, Any]]:
        """
        Fetch every document in the collection, text and metadata only.

        Chroma is the authoritative copy of the chunk text, so this can rebuild
        derived indexes (e.g. BM25) without re-ingesting or re-embedding.

        Returns:
            List of documents with text and metadata
        """
        result = self.chroma._collection.get(include=["documents", "metadatas"])
        docs = [
            {"text": text, "metadata": meta or {}}
            for text, meta in zip(
                result.get("documents") or [], result.get("metadatas") or [], strict=True
            )
        ]
        self.logger.info(f"Fetched {len(docs)} documents from '{self.collection_name}'")
        return docs

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.logger.warning("Clearing entire collection")
        self.chroma.delete_collection()
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.chroma._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model_name,
        }

    @staticmethod
    def list_collections(persist_directory: str = None) -> list[str]:
        """
        Return all collection names stored in the ChromaDB at persist_directory.

        Args:
            persist_directory: Path to the ChromaDB directory (defaults to Config.PERSIST_DIRECTORY)

        Returns:
            Sorted list of collection name strings
        """
        import chromadb

        path = persist_directory or Config.PERSIST_DIRECTORY
        client = chromadb.PersistentClient(path=path)
        return sorted(c.name for c in client.list_collections())

    @staticmethod
    def delete_collection(collection_name: str, persist_directory: str = None) -> None:
        """
        Permanently delete a named collection from ChromaDB.

        Args:
            collection_name: Name of the collection to delete
            persist_directory: Path to ChromaDB directory (defaults to Config.PERSIST_DIRECTORY)
        """
        import chromadb

        path = persist_directory or Config.PERSIST_DIRECTORY
        client = chromadb.PersistentClient(path=path)
        client.delete_collection(collection_name)


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()

    sample_docs = [
        {
            "text": "Q4 revenue increased by 25% compared to last year",
            "metadata": {"source": "finance_report", "category": "revenue"},
        },
        {
            "text": "Daily active users reached 1 million milestone",
            "metadata": {"source": "analytics_dashboard", "category": "users"},
        },
        {
            "text": "Customer satisfaction score improved to 4.5 out of 5",
            "metadata": {"source": "customer_feedback", "category": "satisfaction"},
        },
    ]

    vector_store.add_documents(sample_docs)

    results = vector_store.search("How is our revenue performing?", top_k=2)
    print("\nSearch Results:")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']:.4f}")

    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
