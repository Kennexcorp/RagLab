"""
Shared pytest fixtures.
"""

import pytest

from raglab.config import Config


@pytest.fixture(autouse=True)
def isolated_project_root(tmp_path, monkeypatch):
    """Redirect every path the system derives from PROJECT_ROOT into a temp directory.

    Without this, any test constructing a VectorStore or RAGSystem persists Chroma
    collections, BM25 pickles and rag_system.log into the developer's real working tree.
    DATA_DIR, VECTOR_DB_DIR and PERSIST_DIRECTORY are all derived from PROJECT_ROOT and
    read at call time, so patching the one property covers all of them.
    """
    monkeypatch.setattr(type(Config), "PROJECT_ROOT", property(lambda self: tmp_path))
