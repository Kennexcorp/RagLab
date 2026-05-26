"""
Document parser for the RAG system GUI.
Uses LangChain community document loaders to extract text from uploaded files.
Supports PDF, DOCX, TXT, and MD formats.
CSV and JSON are NOT handled here — route those through DataLoader.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentParser:
    """Parse uploaded file bytes to a list of LangChain Documents using LangChain loaders."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    @staticmethod
    def parse(file_bytes: bytes, filename: str) -> str:
        """
        Write bytes to a temp file, load with the appropriate LangChain loader,
        and return the combined plain text.

        Args:
            file_bytes: Raw bytes of the uploaded file
            filename:   Original filename (used to detect file type)

        Returns:
            Plain-text content joined from all pages/sections

        Raises:
            ValueError: If the file type is unsupported or no text could be extracted
        """
        suffix = Path(filename).suffix.lower()

        if suffix in {".csv", ".json"}:
            raise ValueError(
                f"Use DataLoader for {suffix.lstrip('.')} files, not DocumentParser."
            )
        if suffix not in DocumentParser.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(sorted(DocumentParser.SUPPORTED_EXTENSIONS))}"
            )

        # LangChain loaders require a file path, so write to a temp file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs = DocumentParser._load(tmp_path, suffix)
        finally:
            os.unlink(tmp_path)

        text = "\n\n".join(d.page_content for d in docs if d.page_content.strip())

        if not text:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "If this is a scanned PDF, please run OCR on it first."
            )

        return text

    @staticmethod
    def _load(file_path: str, suffix: str) -> List[Document]:
        """Dispatch to the correct LangChain loader by file extension."""
        if suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(file_path).load()

        elif suffix == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(file_path).load()

        elif suffix in {".txt", ".md"}:
            from langchain_community.document_loaders import TextLoader
            # autodetect encoding; fall back to latin-1 if UTF-8 fails
            try:
                return TextLoader(file_path, encoding="utf-8").load()
            except UnicodeDecodeError:
                return TextLoader(file_path, encoding="latin-1").load()
