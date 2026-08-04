"""
Tests for BM25 index recovery and fallback reporting.
"""

from raglab.retrieval.retriever import Retriever, bm25_index_path
from raglab.retrieval.vector_store import VectorStore

DOCS = [
    {"text": "Q4 revenue increased by 25% to $10M", "metadata": {"source": "report1"}},
    {"text": "User engagement shows 1M daily active users", "metadata": {"source": "report2"}},
    {"text": "Customer satisfaction score is 4.5 out of 5", "metadata": {"source": "report3"}},
]


def _retriever(collection="test-bm25-recovery"):
    vs = VectorStore(collection_name=collection, log_level="ERROR")
    vs.add_documents(DOCS)
    return Retriever(vector_store=vs, log_level="ERROR")


def test_get_all_documents_returns_full_corpus():
    ret = _retriever("test-bm25-getall")
    docs = ret.vector_store.get_all_documents()

    assert len(docs) == len(DOCS)
    assert {d["text"] for d in docs} == {d["text"] for d in DOCS}
    assert all("metadata" in d for d in docs)


def test_hybrid_rebuilds_missing_index_instead_of_degrading():
    """A collection with documents but no BM25 index must rebuild, not fall back.

    Collections ingested before per-collection BM25 indexes existed have no index
    file, which silently turned every hybrid query into a semantic one.
    """
    ret = _retriever()
    assert not ret.hybrid_retriever.is_fitted, "precondition: no index on disk yet"

    results = ret.retrieve("revenue", top_k=3, strategy="hybrid")

    assert ret.hybrid_retriever.is_fitted, "hybrid should have rebuilt its index"
    assert ret.last_fallback is None, "rebuilding means no fallback should be reported"
    assert len(results) > 0
    assert bm25_index_path(ret.vector_store.collection_name).exists(), "index should persist"


def test_rebuild_covers_documents_added_after_the_first_ingestion():
    """BM25 fitting replaces the index wholesale, so it must be fitted on everything.

    Fitting on only the newly ingested batch dropped earlier documents out of
    keyword search entirely.
    """
    ret = _retriever("test-bm25-incremental")
    ret.fit_hybrid_search(DOCS[:1])  # simulate fitting on one batch only

    ret.vector_store.add_documents(
        [{"text": "Warehouse costs fell 12% in October", "metadata": {"source": "report4"}}]
    )
    ret.rebuild_hybrid_index()

    docs = ret.vector_store.get_all_documents()
    assert len(docs) == len(DOCS) + 1
    assert ret.hybrid_retriever.is_fitted


def test_fallback_is_reported_for_unknown_strategy():
    ret = _retriever("test-bm25-unknown")

    ret.retrieve("revenue", top_k=2, strategy="does-not-exist")

    assert ret.last_fallback is not None
    assert "does-not-exist" in ret.last_fallback


def test_last_fallback_resets_between_queries():
    ret = _retriever("test-bm25-reset")

    ret.retrieve("revenue", top_k=2, strategy="does-not-exist")
    assert ret.last_fallback is not None

    ret.retrieve("revenue", top_k=2, strategy="semantic")
    assert ret.last_fallback is None, "a successful query must clear the previous fallback"
