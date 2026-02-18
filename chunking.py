"""
Text chunking module for RAG system.
Uses LangChain's RecursiveCharacterTextSplitter for robust, production-tested splitting.
"""

import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils import setup_logging


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
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ('fixed' or 'semantic')
            log_level: Logging level
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.logger = setup_logging(log_level)

        # 'semantic' splits on paragraph → sentence → word boundaries;
        # 'fixed' splits on word boundaries only.
        separators = (
            ["\n\n", "\n", ". ", " ", ""]
            if strategy == "semantic"
            else [" ", ""]
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

        self.logger.info(
            f"TextChunker initialized (strategy={strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a document using the configured strategy.

        Args:
            document: Document with 'text' and 'metadata' fields

        Returns:
            List of chunks with metadata (includes chunk_index)
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text:
            self.logger.warning("Empty text in document")
            return []

        lc_doc = Document(page_content=text, metadata=metadata)
        splits = self.splitter.split_documents([lc_doc])

        chunks = [
            {"text": split.page_content, "metadata": {**split.metadata, "chunk_index": i}}
            for i, split in enumerate(splits)
        ]

        self.logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of all chunks from all documents (each chunk has document_index in metadata)
        """
        self.logger.info(f"Chunking {len(documents)} documents")

        all_chunks = []
        for doc_index, document in enumerate(documents):
            for chunk in self.chunk_document(document):
                chunk["metadata"]["document_index"] = doc_index
                all_chunks.append(chunk)

        self.logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks


if __name__ == "__main__":
    # Example usage
    chunker = TextChunker(chunk_size=200, chunk_overlap=20, strategy="semantic")

    sample_doc = {
        "text": """This is the first paragraph about revenue metrics.
        Our Q4 revenue increased by 25% compared to last year.

        This is the second paragraph about user engagement.
        Daily active users have grown significantly.
        We now have over 1 million daily active users.

        This is the third paragraph with conclusions.
        Overall performance has been excellent this quarter.""",
        "metadata": {"source": "Q4_report.pdf", "date": "2025-12-31"},
    }

    chunks = chunker.chunk_document(sample_doc)
    print(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk["text"][:100] + "...")
        print(f"Metadata: {chunk['metadata']}")
