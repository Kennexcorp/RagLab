"""
Retrieval module for RAG system.
Handles query processing and context retrieval from vector store.
Supports eight interchangeable retrieval strategies.
"""

import os
from pathlib import Path
from typing import Any

# langchain_community v0.4+ removed several vectorstore classes that langchain_classic's
# SelfQueryRetriever still tries to import. Stub them all out so imports never fail.
import langchain_community.vectorstores as _lcvs

for _cls in [
    "DatabricksVectorSearch",
    "DeepLake",
    "Milvus",
    "Neo4jVector",
    "Qdrant",
    "Weaviate",
    "MongoDBAtlasVectorSearch",
    "TencentVectorDb",
    "Pinecone",
]:
    if not hasattr(_lcvs, _cls):
        setattr(_lcvs, _cls, type(_cls, (), {}))
del _cls

from raglab.config import Config  # noqa: E402
from raglab.models import ContextBundle, RetrievedChunk  # noqa: E402
from raglab.retrieval.hybrid_search import LangChainHybridRetriever  # noqa: E402
from raglab.retrieval.vector_store import VectorStore  # noqa: E402
from raglab.utils import count_tokens, format_context_for_llm, setup_logging, timer  # noqa: E402

# ---------------------------------------------------------------------------
# Strategy registry (mirrors chunking.py's STRATEGIES / STRATEGY_INFO)
# ---------------------------------------------------------------------------

RETRIEVAL_STRATEGIES = [
    "semantic",
    "hybrid",
    "mmr",
    "multi-query",
    "reranking",
    "hyde",
    "self-query",
    "parent-child",
]

RETRIEVAL_STRATEGY_INFO: dict[str, dict[str, str]] = {
    "semantic": {
        "label": "Semantic (Default)",
        "description": (
            "Pure vector similarity search using HuggingFace embeddings. "
            "Finds documents that are conceptually similar to the query. "
            "Best general-purpose choice."
        ),
    },
    "hybrid": {
        "label": "Hybrid (BM25 + Semantic)",
        "description": (
            "Combines keyword (BM25) and vector similarity via weighted fusion. "
            "Better for queries mixing exact terms and concepts. "
            "Use the semantic weight slider to tune the balance."
        ),
    },
    "mmr": {
        "label": "MMR – Diverse Results",
        "description": (
            "Maximal Marginal Relevance balances relevance with diversity. "
            "Prevents the LLM receiving five near-identical chunks. "
            "Best when documents have repeated or overlapping sections."
        ),
    },
    "multi-query": {
        "label": "Multi-Query",
        "description": (
            "Uses the LLM to generate 3–5 rephrased versions of your question, "
            "retrieves for each, then deduplicates. "
            "Improves recall when the user's phrasing doesn't match the document's vocabulary. "
            "Requires a configured LLM."
        ),
    },
    "reranking": {
        "label": "Reranking (Cross-Encoder)",
        "description": (
            "Retrieves a wide candidate set then re-scores with a cross-encoder model "
            "for precise relevance ranking. Highest precision of all strategies. "
            "Slightly slower due to the second-pass model scoring each candidate."
        ),
    },
    "hyde": {
        "label": "HyDE – Hypothetical Document",
        "description": (
            "Generates a hypothetical ideal answer with the LLM, embeds it, and searches "
            "for real documents similar to that embedding. "
            "Bridges question/answer vocabulary gaps — useful when queries are phrased "
            "very differently from the document text. Requires a configured LLM."
        ),
    },
    "self-query": {
        "label": "Self-Query (Metadata Filters)",
        "description": (
            "The LLM parses metadata filters directly from your question "
            "(e.g. 'legal documents by Jane' → category=legal, author=Jane). "
            "Combines semantic search with precise structured filtering. "
            "Falls back to semantic search if no filters are detected. Requires a configured LLM."
        ),
    },
    "parent-child": {
        "label": "Parent-Child Chunking",
        "description": (
            "Indexes small child chunks for precise retrieval but returns their larger "
            "parent chunks to the LLM for richer context. "
            "Best for dense documents where the answer spans more than one small chunk. "
            "Re-ingest documents after switching to this strategy."
        ),
    },
}


class Retriever:
    """Retrieve relevant context for queries using configurable search strategies."""

    def __init__(self, vector_store: VectorStore = None, log_level: str = "INFO"):
        self.logger = setup_logging(log_level)
        self.log_level = log_level
        self.vector_store = vector_store or VectorStore(log_level=log_level)
        self.top_k = Config.TOP_K_RESULTS
        self.similarity_threshold = Config.SIMILARITY_THRESHOLD
        self.max_context_length = Config.MAX_CONTEXT_LENGTH
        self.strategy = Config.RETRIEVAL_STRATEGY

        # Hybrid BM25 component — always initialised so "hybrid" strategy works at any time
        self.use_hybrid = Config.USE_HYBRID_SEARCH
        if self.use_hybrid:
            self.hybrid_retriever = LangChainHybridRetriever(
                semantic_weight=Config.SEMANTIC_WEIGHT,
                keyword_weight=Config.KEYWORD_WEIGHT,
                log_level=log_level,
            )
            self.logger.info(
                f"Hybrid search ready "
                f"(semantic: {Config.SEMANTIC_WEIGHT:.0%}, keyword: {Config.KEYWORD_WEIGHT:.0%})"
            )
        else:
            self.hybrid_retriever = None

        if self.use_hybrid and self.hybrid_retriever:
            self.load_hybrid_index()

        # Parent-child doc store — created lazily in _retrieve_parent_child
        self._parent_child_retriever = None

    # ------------------------------------------------------------------
    # Hybrid index management (unchanged)
    # ------------------------------------------------------------------

    def fit_hybrid_search(self, documents: list[dict[str, Any]]):
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.fit(documents)
            self.save_hybrid_index()
            self.logger.info("Hybrid search fitted on documents")

    def save_hybrid_index(self):
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.save_index(Config.BM25_INDEX_PATH)

    def load_hybrid_index(self):
        if self.use_hybrid and self.hybrid_retriever:
            self.hybrid_retriever.load_index(Config.BM25_INDEX_PATH)

    # ------------------------------------------------------------------
    # LLM factory (used by multi-query, hyde, self-query)
    # ------------------------------------------------------------------

    def _build_llm(self, provider: str = None, model: str = None, api_key: str = None):
        """Build a bare LangChain LLM for retrieval-time use."""
        provider = provider or Config.LLM_PROVIDER
        model = model or Config.LLM_MODEL
        api_key = api_key or Config.get_llm_api_key()

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=model, temperature=0, api_key=api_key)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=model, temperature=0, api_key=api_key)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model, temperature=0)
        else:
            raise ValueError(f"Unsupported provider for retrieval LLM: {provider}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        semantic_results = self.vector_store.search(
            query=query, top_k=top_k, metadata_filter=metadata_filter
        )
        results = []
        for result in semantic_results:
            distance = result.get("distance", 0)
            # ChromaDB returns L2 distances. For L2-normalised embeddings
            # (sentence-transformers always normalises), the cosine similarity
            # is: cos_sim = 1 - d² / 2  (derived from ||a-b||² = 2 - 2·cos).
            # Clamp to [0, 1] to handle tiny floating-point overshoots.
            similarity = max(0.0, min(1.0, 1.0 - (distance**2) / 2.0))
            result["similarity_score"] = similarity
            if similarity >= self.similarity_threshold:
                results.append(result)
        return results

    def _retrieve_hybrid(
        self, query: str, top_k: int, semantic_weight: float = None
    ) -> list[dict[str, Any]]:
        if not (self.hybrid_retriever and self.hybrid_retriever.is_fitted):
            self.logger.warning("Hybrid retriever not fitted — falling back to semantic")
            return self._retrieve_semantic(query, top_k)

        if semantic_weight is not None and self.hybrid_retriever.ensemble_retriever is not None:
            kw = round(1.0 - semantic_weight, 4)
            self.hybrid_retriever.ensemble_retriever.weights = [semantic_weight, kw]

        vector_retriever = self.vector_store.as_retriever(k=20)
        results = self.hybrid_retriever.search(query, vector_retriever, top_k=top_k)
        for i, result in enumerate(results):
            result["similarity_score"] = 1.0 / (i + 1)
        return results

    def _retrieve_mmr(self, query: str, top_k: int) -> list[dict[str, Any]]:
        retriever = self.vector_store.as_retriever(k=top_k, search_type="mmr")
        docs = retriever.invoke(query)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1.0 / (i + 1),
            }
            for i, doc in enumerate(docs)
        ]

    def _retrieve_multi_query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        base_retriever = self.vector_store.as_retriever(k=top_k)
        llm = self._build_llm()
        mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        docs = mq_retriever.invoke(query)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1.0 / (i + 1),
            }
            for i, doc in enumerate(docs[:top_k])
        ]

    def _retrieve_reranking(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "Reranking strategy requires: pip install sentence-transformers"
            ) from e

        # Bypass the similarity threshold — fetch raw candidates directly from the
        # vector store so the cross-encoder has a full pool to rerank.
        raw = self.vector_store.search(query=query, top_k=top_k * 3)
        candidates = [{**doc, "similarity_score": 1 / (1 + doc.get("distance", 0))} for doc in raw]
        if not candidates:
            return []

        ce_model = CrossEncoder(Config.RERANKER_MODEL)
        pairs = [(query, doc["text"]) for doc in candidates]
        scores = ce_model.predict(pairs)

        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
        return [{**doc, "similarity_score": float(score)} for doc, score in ranked]

    def _retrieve_hyde(self, query: str, top_k: int) -> list[dict[str, Any]]:
        from langchain_core.prompts import ChatPromptTemplate

        llm = self._build_llm()
        hyde_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Write a short hypothetical document passage "
                    "that would directly answer the following question. "
                    "Write only the passage, no preamble.",
                ),
                ("human", "{question}"),
            ]
        )
        chain = hyde_prompt | llm
        hypothetical_doc = chain.invoke({"question": query}).content
        self.logger.debug(f"HyDE hypothetical doc: {hypothetical_doc[:100]}…")

        # Search using the hypothetical document text, bypassing the similarity threshold —
        # the hypothetical embedding sits further from real docs by design.
        raw = self.vector_store.search(query=hypothetical_doc, top_k=top_k)
        return [{**doc, "similarity_score": 1 / (1 + doc.get("distance", 0))} for doc in raw]

    def _retrieve_self_query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            from langchain_classic.chains.query_constructor.schema import AttributeInfo
            from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

            # Pass ChromaTranslator explicitly — avoids langchain_classic's _get_builtin_translator
            # which tries to import optional vectorstores (e.g. DatabricksVectorSearch) that may
            # not be present in the installed langchain_community version.
            from langchain_community.query_constructors.chroma import ChromaTranslator
        except ImportError as exc:
            self.logger.warning(f"Self-query imports failed ({exc}), falling back to semantic")
            return self._retrieve_semantic(query, top_k)

        metadata_field_info = [
            AttributeInfo(name="title", description="Title of the document", type="string"),
            AttributeInfo(
                name="category",
                description="Category or topic of the document",
                type="string",
            ),
            AttributeInfo(
                name="source",
                description="Source or filename of the document",
                type="string",
            ),
            AttributeInfo(name="author", description="Author of the document", type="string"),
            AttributeInfo(
                name="tags",
                description="Comma-separated tags for the document",
                type="string",
            ),
        ]
        llm = self._build_llm()
        try:
            sq_retriever = SelfQueryRetriever.from_llm(
                llm=llm,
                vectorstore=self.vector_store.chroma,
                document_contents="Passages from uploaded documents",
                metadata_field_info=metadata_field_info,
                structured_query_translator=ChromaTranslator(),
                verbose=False,
            )
            docs = sq_retriever.invoke(query)
        except Exception as exc:
            self.logger.warning(f"Self-query failed ({exc}), falling back to semantic")
            return self._retrieve_semantic(query, top_k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1.0 / (i + 1),
            }
            for i, doc in enumerate(docs[:top_k])
        ]

    def _retrieve_parent_child(self, query: str, top_k: int) -> list[dict[str, Any]]:
        from langchain_classic.retrievers.parent_document_retriever import (
            ParentDocumentRetriever,
        )
        from langchain_classic.storage import LocalFileStore
        from langchain_classic.storage._lc_store import create_kv_docstore
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if self._parent_child_retriever is None:
            store_path = os.path.join(
                Config.PERSIST_DIRECTORY,
                "parent_docs",
                self.vector_store.collection_name,
            )
            Path(store_path).mkdir(parents=True, exist_ok=True)
            fs = LocalFileStore(store_path)
            docstore = create_kv_docstore(fs)
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            )
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.PARENT_CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
            )
            self._parent_child_retriever = ParentDocumentRetriever(
                vectorstore=self.vector_store.chroma,
                docstore=docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )

        try:
            docs = self._parent_child_retriever.invoke(query)
        except Exception as exc:
            self.logger.warning(f"Parent-child retrieval failed ({exc}), falling back to semantic")
            return self._retrieve_semantic(query, top_k)
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": 1.0 / (i + 1),
            }
            for i, doc in enumerate(docs[:top_k])
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @timer
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: dict[str, Any] | None = None,
        strategy: str = None,
        semantic_weight: float = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant documents for a query using the selected strategy.

        Args:
            query:          User query
            top_k:          Number of results to retrieve
            metadata_filter: Optional metadata filter (semantic strategy only)
            strategy:       Override retrieval strategy (None uses instance default)
            semantic_weight: Hybrid strategy only — semantic vs keyword balance

        Returns:
            List of relevant documents
        """
        top_k = top_k or self.top_k
        strategy = strategy or self.strategy

        self.logger.info(f"Retrieving for: '{query}' (strategy={strategy}, top_k={top_k})")

        if strategy == "semantic":
            results = self._retrieve_semantic(query, top_k, metadata_filter)
        elif strategy == "hybrid":
            results = self._retrieve_hybrid(query, top_k, semantic_weight)
        elif strategy == "mmr":
            results = self._retrieve_mmr(query, top_k)
        elif strategy == "multi-query":
            results = self._retrieve_multi_query(query, top_k)
        elif strategy == "reranking":
            results = self._retrieve_reranking(query, top_k)
        elif strategy == "hyde":
            results = self._retrieve_hyde(query, top_k)
        elif strategy == "self-query":
            results = self._retrieve_self_query(query, top_k)
        elif strategy == "parent-child":
            results = self._retrieve_parent_child(query, top_k)
        else:
            self.logger.warning(f"Unknown strategy '{strategy}', falling back to semantic")
            results = self._retrieve_semantic(query, top_k, metadata_filter)

        return [RetrievedChunk(**doc).model_dump() for doc in results]

    def build_context(
        self,
        query: str,
        top_k: int = None,
        strategy: str = None,
        semantic_weight: float = None,
    ) -> dict[str, Any]:
        """
        Build context for LLM generation.

        Args:
            query:          User query
            top_k:          Number of documents to retrieve
            strategy:       Override retrieval strategy
            semantic_weight: Hybrid strategy weight override

        Returns:
            Dictionary with context string and source documents
        """
        documents = self.retrieve(
            query, top_k=top_k, strategy=strategy, semantic_weight=semantic_weight
        )

        if not documents:
            self.logger.warning("No relevant documents found")
            return ContextBundle().model_dump()

        context = format_context_for_llm(documents)

        context_tokens = count_tokens(context)
        if context_tokens > self.max_context_length:
            self.logger.warning(
                f"Context too long ({context_tokens} tokens), "
                f"truncating to {self.max_context_length}"
            )
            while context_tokens > self.max_context_length and len(documents) > 1:
                documents.pop()
                context = format_context_for_llm(documents)
                context_tokens = count_tokens(context)

        return ContextBundle(
            context=context,
            sources=documents,
            num_sources=len(documents),
            total_tokens=context_tokens,
        ).model_dump()

    def retrieve_by_metadata(
        self, metadata_filter: dict[str, Any], top_k: int = None
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.top_k
        self.logger.info(f"Retrieving by metadata: {metadata_filter}")
        return self.vector_store.get_by_metadata(metadata_filter, top_k=top_k)
