"""
Shared pydantic models for the RAG system.
Used for construction-time validation and to guarantee consistent response shapes.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentMetadata(BaseModel):
    """Metadata for a document ingested via ingest_document_with_metadata."""

    title: str
    category: str
    source: str
    description: str = ""
    tags: str = ""
    author: str = ""
    type: str = "uploaded_document"
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    record_index: int = 0
    id: str | None = None
    timestamp: str | None = None
    date: str | None = None

    @field_validator("title", "category", "source")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class RetrievedChunk(BaseModel):
    """A single retrieved document chunk."""

    text: str
    metadata: dict[str, Any] = {}
    similarity_score: float | None = None
    distance: float | None = None


class ContextBundle(BaseModel):
    """Context assembled from retrieved chunks for LLM generation."""

    context: str = ""
    sources: list[RetrievedChunk] = []
    num_sources: int = 0
    total_tokens: int = 0


class QueryResponse(BaseModel):
    """Response returned by Generator.generate / RAGSystem.query."""

    answer: str
    question: str | None = None
    search_question: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    finish_reason: str | None = None
    prompt_sent: str | None = None
    sources: list[RetrievedChunk] = []
    num_sources: int = 0
    answer_with_citations: str | None = None
    context_tokens: int | None = None
    performance: dict[str, Any] | None = None
