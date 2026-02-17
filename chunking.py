"""
Text chunking module for RAG system.
Implements various chunking strategies to split documents into manageable pieces.
"""

import logging
from typing import List, Dict, Any
import re

from utils import setup_logging, count_tokens


class TextChunker:
    """Split text into chunks for embedding and retrieval."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "fixed",
        log_level: str = "INFO",
    ):
        """
        Initialize TextChunker.

        Args:
            chunk_size: Target size of each chunk in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            strategy: Chunking strategy ('fixed' or 'semantic')
            log_level: Logging level
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.logger = setup_logging(log_level)

    def chunk_by_tokens(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by fixed token count.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        # Split text into words (simple tokenization)
        words = text.split()
        chunks = []

        # Calculate approximate words per chunk (rough estimate: 1 token ≈ 0.75 words)
        words_per_chunk = int(self.chunk_size * 0.75)
        overlap_words = int(self.chunk_overlap * 0.75)

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Create chunk with metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["chunk_start"] = start
            chunk_metadata["chunk_end"] = min(end, len(words))
            chunk_metadata["total_words"] = len(words)

            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

            # Move start position with overlap
            start = end - overlap_words
            chunk_index += 1

            # Prevent infinite loop
            if start >= len(words) or end >= len(words):
                break

        return chunks

    def chunk_by_sentences(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by sentences, respecting token limits.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        # Split into sentences (simple regex-based splitting)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)

            # If adding this sentence would exceed chunk size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)

                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["num_sentences"] = len(current_chunk)

                chunks.append({"text": chunk_text, "metadata": chunk_metadata})

                # Start new chunk with overlap (keep last few sentences)
                overlap_sentences = int(
                    len(current_chunk) * (self.chunk_overlap / self.chunk_size)
                )
                current_chunk = (
                    current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
                )
                current_tokens = sum(count_tokens(s) for s in current_chunk)
                chunk_index += 1

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["num_sentences"] = len(current_chunk)

            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def chunk_by_paragraphs(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by paragraphs, respecting token limits.

        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk

        Returns:
            List of chunks with metadata
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)

            # If paragraph alone exceeds chunk size, split it by sentences
            if paragraph_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["chunk_index"] = chunk_index
                    chunks.append({"text": chunk_text, "metadata": chunk_metadata})
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split large paragraph by sentences
                para_chunks = self.chunk_by_sentences(paragraph, metadata)
                chunks.extend(para_chunks)
                continue

            # If adding this paragraph would exceed chunk size, save current chunk
            if current_tokens + paragraph_tokens > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append({"text": chunk_text, "metadata": chunk_metadata})
                chunk_index += 1
                current_chunk = []
                current_tokens = 0

            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        return chunks

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using the configured strategy.

        Args:
            document: Document with 'text' and 'metadata' fields

        Returns:
            List of chunks
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text:
            self.logger.warning("Empty text in document")
            return []

        # Choose chunking strategy
        if self.strategy == "semantic":
            # Semantic chunking: try paragraphs first, then sentences
            chunks = self.chunk_by_paragraphs(text, metadata)
        else:
            # Fixed chunking: simple token-based splitting
            chunks = self.chunk_by_tokens(text, metadata)

        self.logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks from all documents
        """
        self.logger.info(f"Chunking {len(documents)} documents")

        all_chunks = []
        for doc_index, document in enumerate(documents):
            chunks = self.chunk_document(document)

            # Add document index to metadata
            for chunk in chunks:
                chunk["metadata"]["document_index"] = doc_index

            all_chunks.extend(chunks)

        self.logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks

if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="semantic")

    sample_doc = {
        "text": """This is the first paragraph about revenue metrics. 
        Our Q4 revenue increased by 25% compared to last year.
        
        This is the second paragraph about user engagement. 
        Daily active users have grown significantly. 
        We now have over 1 million daily active users.
        
        This is the third paragraph with conclusions. 
        Overall performance has been excellent this quarter.""",
        "metadata": {"source": "Q4_report.pdf", "date": "2025-12-31"}
    }

    chunks = chunker.chunk_document(sample_doc)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk["text"][:100] + "...")
        print(f"Metadata: {chunk['metadata']}")
