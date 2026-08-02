"""
Unit tests for chunking module.
"""

import pytest

from raglab.ingestion.chunking import TextChunker


class TestTextChunker:
    """Test cases for TextChunker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="fixed")

    def test_chunk_by_tokens(self):
        """Test token-based chunking."""
        document = {
            "text": "This is a test document. " * 50,  # Long text
            "metadata": {"source": "test"},
        }

        chunks = self.chunker.chunk_document(document)

        assert len(chunks) > 1, "Should create multiple chunks"
        assert all("text" in chunk for chunk in chunks), "All chunks should have text"
        assert all("metadata" in chunk for chunk in chunks), "All chunks should have metadata"

    def test_chunk_by_sentences(self):
        """Test sentence-based chunking."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, strategy="semantic")

        document = {
            "text": (
                "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
            ),
            "metadata": {"source": "test"},
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) >= 1, "Should create at least one chunk"
        for chunk in chunks:
            assert "chunk_index" in chunk["metadata"], "Should have chunk index"

    def test_empty_document(self):
        """Test handling of empty document."""
        document = {"text": "", "metadata": {"source": "test"}}

        chunks = self.chunker.chunk_document(document)

        assert len(chunks) == 0, "Empty document should produce no chunks"

    def test_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        document = {
            "text": "Test document with metadata.",
            "metadata": {"source": "test", "category": "example"},
        }

        chunks = self.chunker.chunk_document(document)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "test", "Should preserve source"
            assert chunk["metadata"]["category"] == "example", "Should preserve category"

    def test_chunk_documents_batch(self):
        """Test chunking multiple documents."""
        documents = [
            {"text": "Document 1 text.", "metadata": {"id": 1}},
            {"text": "Document 2 text.", "metadata": {"id": 2}},
            {"text": "Document 3 text.", "metadata": {"id": 3}},
        ]

        chunks = self.chunker.chunk_documents(documents)

        assert len(chunks) >= 3, "Should create at least one chunk per document"

        # Check document indices are added
        doc_indices = set(chunk["metadata"]["document_index"] for chunk in chunks)
        assert len(doc_indices) == 3, "Should have chunks from all 3 documents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
