"""
Vector store module for RAG system.
Handles embedding generation and vector database operations using ChromaDB.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import Config
from utils import setup_logging, timer, retry_on_error


class VectorStore:
    """Manage vector embeddings and similarity search using ChromaDB."""

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
            collection_name: Name of the collection
            embedding_model: Name of the embedding model
            persist_directory: Directory to persist the database
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)

        # Use config values as defaults
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL
        self.persist_directory = persist_directory or Config.PERSIST_DIRECTORY

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB client
        self.logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Dashboard data for RAG system"},
        )

        self.logger.info(
            f"Vector store initialized with collection: {self.collection_name}"
        )

    @timer
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        self.logger.debug(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    @retry_on_error(max_retries=3, delay=1.0)
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents with 'text' and 'metadata' fields
        """
        if not documents:
            self.logger.warning("No documents to add")
            return

        self.logger.info(f"Adding {len(documents)} documents to vector store")

        # Extract texts and metadata
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # Generate IDs
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Convert metadata values to strings (ChromaDB requirement)
        processed_metadatas = []
        for metadata in metadatas:
            processed_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    processed_metadata[key] = str(value)
            processed_metadatas.append(processed_metadata)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=processed_metadatas,
        )

        self.logger.info(f"Successfully added {len(documents)} documents")

    @timer
    def search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Query text
            top_k: Number of results to return
            metadata_filter: Optional metadata filter

        Returns:
            List of search results with text, metadata, and similarity scores
        """
        top_k = top_k or Config.TOP_K_RESULTS

        self.logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.generate_embeddings([query])[0]

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=metadata_filter
        )

        # Format results
        formatted_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        self.logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.

        Args:
            doc_id: Document ID

        Returns:
            Document data or None if not found
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "text": result["documents"][0],
                    "metadata": result["metadatas"][0],
                }
        except Exception as e:
            self.logger.error(f"Error retrieving document {doc_id}: {str(e)}")

        return None

    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            doc_ids: List of document IDs to delete
        """
        self.logger.info(f"Deleting {len(doc_ids)} documents")
        self.collection.delete(ids=doc_ids)

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.logger.warning("Clearing entire collection")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Dashboard data for RAG system"},
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model_name,
        }


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()

    # Sample documents
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

    # Add documents
    vector_store.add_documents(sample_docs)

    # Search
    results = vector_store.search("How is our revenue performing?", top_k=2)

    print("\nSearch Results:")
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")

    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\nCollection Stats: {stats}")
