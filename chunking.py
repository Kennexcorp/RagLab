"""
Text chunking module for RAG system.
Supports multiple LangChain splitting strategies.
"""

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config
from utils import setup_logging


# All supported strategy identifiers
STRATEGIES = [
    "semantic",
    "fixed",
    "markdown",
    "markdown-headers",
    "token",
    "python",
    "latex",
    "spacy",
    "nltk",
    "semantic-embedding",
]

# Human-readable labels and descriptions shown in the GUI
STRATEGY_INFO: Dict[str, Dict[str, str]] = {
    "semantic": {
        "label": "Semantic (Recommended)",
        "description": (
            "Splits on paragraph → sentence → word boundaries in that order. "
            "Best for reports, policies, and articles where sentence coherence matters."
        ),
    },
    "fixed": {
        "label": "Fixed (Word Boundaries)",
        "description": (
            "Splits on word boundaries only, ignoring sentence/paragraph structure. "
            "Best for CSV-derived text, logs, or data with no natural paragraph breaks."
        ),
    },
    "markdown": {
        "label": "Markdown",
        "description": (
            "Respects markdown syntax — preserves code blocks, lists, and inline "
            "formatting across chunk boundaries."
        ),
    },
    "markdown-headers": {
        "label": "Markdown Headers",
        "description": (
            "Splits on heading hierarchy (#, ##, ###). Each chunk carries its section "
            "heading as metadata, which improves retrieval context significantly."
        ),
    },
    "token": {
        "label": "Token-aware",
        "description": (
            "Splits by token count (via tiktoken) rather than characters. "
            "Use when querying models with strict token context windows — "
            "chunk size means tokens here, not characters."
        ),
    },
    "python": {
        "label": "Python Code",
        "description": (
            "Splits on class, function, and method boundaries. "
            "Keeps code units intact for better code search and retrieval."
        ),
    },
    "latex": {
        "label": "LaTeX / Academic",
        "description": (
            "Splits on LaTeX structural commands (\\section, \\chapter, \\paragraph). "
            "Best for academic PDFs that have been converted to LaTeX source."
        ),
    },
    "spacy": {
        "label": "spaCy Sentences",
        "description": (
            "Uses spaCy's sentence detector for high-quality sentence boundaries. "
            "More accurate than rule-based splitting. "
            "Requires: pip install spacy && python -m spacy download en_core_web_sm"
        ),
    },
    "nltk": {
        "label": "NLTK Sentences",
        "description": (
            "Uses NLTK's Punkt sentence tokeniser for sentence-aware splitting. "
            "Similar to spaCy but lighter. "
            "Requires: pip install nltk (punkt data downloaded automatically)."
        ),
    },
    "semantic-embedding": {
        "label": "Semantic Embedding (Highest Quality)",
        "description": (
            "Splits where meaning changes by comparing embedding similarity between "
            "adjacent sentences. Produces variable-length, semantically coherent chunks. "
            "Highest retrieval quality but slowest. "
            "Requires: pip install langchain-experimental"
        ),
    },
}


class TextChunker:
    """Split text into chunks using a configurable LangChain splitting strategy."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "semantic",
        embedding_model_name: str = None,
        log_level: str = "INFO",
    ):
        """
        Args:
            chunk_size:           Target chunk size (chars for most strategies, tokens for 'token')
            chunk_overlap:        Overlap between consecutive chunks
            strategy:             One of the keys in STRATEGIES
            embedding_model_name: HuggingFace model name — only used by 'semantic-embedding'
            log_level:            Logging level
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.embedding_model_name = embedding_model_name or Config.EMBEDDING_MODEL
        self.logger = setup_logging(log_level)

        self.splitter = self._build_splitter()
        # MarkdownHeaderTextSplitter has a different API — flag it for chunk_document
        self._is_header_splitter = (strategy == "markdown-headers")

        self.logger.info(
            f"TextChunker initialised (strategy={strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap})"
        )

    def _build_splitter(self):
        """Instantiate the correct LangChain splitter for the selected strategy."""
        s = self.strategy
        sz = self.chunk_size
        ov = self.chunk_overlap

        if s == "semantic":
            return RecursiveCharacterTextSplitter(
                chunk_size=sz, chunk_overlap=ov,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )

        elif s == "fixed":
            return RecursiveCharacterTextSplitter(
                chunk_size=sz, chunk_overlap=ov,
                separators=[" ", ""],
                length_function=len,
            )

        elif s == "markdown":
            from langchain_text_splitters import MarkdownTextSplitter
            return MarkdownTextSplitter(chunk_size=sz, chunk_overlap=ov)

        elif s == "markdown-headers":
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4"),
                ],
                strip_headers=False,
            )

        elif s == "token":
            from langchain_text_splitters import TokenTextSplitter
            return TokenTextSplitter(chunk_size=sz, chunk_overlap=ov)

        elif s == "python":
            from langchain_text_splitters import PythonCodeTextSplitter
            return PythonCodeTextSplitter(chunk_size=sz, chunk_overlap=ov)

        elif s == "latex":
            from langchain_text_splitters import LatexTextSplitter
            return LatexTextSplitter(chunk_size=sz, chunk_overlap=ov)

        elif s == "spacy":
            try:
                from langchain_text_splitters import SpacyTextSplitter
                return SpacyTextSplitter(chunk_size=sz, pipeline="en_core_web_sm")
            except (ImportError, OSError) as e:
                raise ImportError(
                    "spaCy strategy requires: "
                    "pip install spacy && python -m spacy download en_core_web_sm"
                ) from e

        elif s == "nltk":
            try:
                import nltk
                nltk.download("punkt_tab", quiet=True)
                from langchain_text_splitters import NLTKTextSplitter
                return NLTKTextSplitter(chunk_size=sz)
            except ImportError as e:
                raise ImportError(
                    "NLTK strategy requires: pip install nltk"
                ) from e

        elif s == "semantic-embedding":
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                return SemanticChunker(embeddings)
            except ImportError as e:
                raise ImportError(
                    "Semantic-embedding strategy requires: pip install langchain-experimental"
                ) from e

        else:
            raise ValueError(
                f"Unknown strategy '{s}'. Choose from: {', '.join(STRATEGIES)}"
            )

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document.

        Args:
            document: Dict with 'text' and 'metadata' keys

        Returns:
            List of chunk dicts with 'text' and 'metadata' (includes chunk_index)
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text:
            self.logger.warning("Empty text in document, skipping")
            return []

        if self._is_header_splitter:
            # MarkdownHeaderTextSplitter.split_text() returns List[Document]
            # and injects header keys into each document's metadata.
            splits = self.splitter.split_text(text)
        else:
            lc_doc = Document(page_content=text, metadata=metadata)
            splits = self.splitter.split_documents([lc_doc])

        chunks = [
            {
                "text": split.page_content,
                "metadata": {
                    **metadata,
                    **split.metadata,   # header splitter adds h1/h2/h3 keys here
                    "chunk_index": i,
                },
            }
            for i, split in enumerate(splits)
        ]

        self.logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents.

        Returns:
            All chunks from all documents, each with document_index in metadata
        """
        self.logger.info(f"Chunking {len(documents)} documents with strategy '{self.strategy}'")
        all_chunks = []
        for doc_index, document in enumerate(documents):
            for chunk in self.chunk_document(document):
                chunk["metadata"]["document_index"] = doc_index
                all_chunks.append(chunk)
        self.logger.info(f"Created {len(all_chunks)} total chunks")
        return all_chunks
