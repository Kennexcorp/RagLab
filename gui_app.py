"""
RAG Document Manager — Streamlit GUI
Manage collections, upload documents, and query your knowledge base.

Run with:
    streamlit run gui_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from rag_system import RAGSystem
from document_parser import DocumentParser
from config import Config
from chunking import STRATEGIES, STRATEGY_INFO
from retriever import RETRIEVAL_STRATEGIES, RETRIEVAL_STRATEGY_INFO


# ---------------------------------------------------------------------------
# Provider → sensible model defaults
# ---------------------------------------------------------------------------
_PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
}


# ---------------------------------------------------------------------------
# Cached RAGSystem — shared across all sessions in this process.
# The embedding model is loaded once and reused for every collection.
# ---------------------------------------------------------------------------
@st.cache_resource
def get_rag_system() -> RAGSystem:
    return RAGSystem(log_level="WARNING")


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_state(_rag: RAGSystem) -> None:
    """Initialise session state keys on first load."""
    if "collection" not in st.session_state:
        st.session_state["collection"] = Config.COLLECTION_NAME  # default: "documents"
    if "llm_provider" not in st.session_state:
        st.session_state["llm_provider"] = Config.LLM_PROVIDER
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = Config.LLM_MODEL
    if "llm_api_key" not in st.session_state:
        st.session_state["llm_api_key"] = ""
    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = False
    if "upload_form_key" not in st.session_state:
        st.session_state["upload_form_key"] = 0
    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = Config.CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state["chunk_overlap"] = Config.CHUNK_OVERLAP
    if "chunk_strategy" not in st.session_state:
        st.session_state["chunk_strategy"] = Config.CHUNKING_STRATEGY
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = Config.LLM_TEMPERATURE
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = Config.SYSTEM_PROMPT
    if "top_k_default" not in st.session_state:
        st.session_state["top_k_default"] = Config.TOP_K_RESULTS
    if "retrieval_strategy" not in st.session_state:
        st.session_state["retrieval_strategy"] = Config.RETRIEVAL_STRATEGY
    if "semantic_weight" not in st.session_state:
        st.session_state["semantic_weight"] = float(Config.SEMANTIC_WEIGHT)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}  # {collection_name: [{"role":..., "content":..., "sources":..., "perf":...}]}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(rag: RAGSystem) -> None:
    with st.sidebar:
        st.title("RAG Manager")

        # ---- Collections section ----
        st.subheader("Collections")

        _all = rag.list_collections()
        # "documents" always first, remaining collections sorted alphabetically
        _default = Config.COLLECTION_NAME
        _others = sorted(c for c in _all if c != _default)
        collections = [_default] + _others if _default in _all else ([_default] + _others or [_default])

        # Ensure current selection is present (e.g. a just-created collection not yet persisted)
        if st.session_state["collection"] not in collections:
            collections.append(st.session_state["collection"])

        selected = st.selectbox(
            "Active collection",
            options=collections,
            index=collections.index(st.session_state["collection"]),
            key="_col_select",
            label_visibility="collapsed",
        )
        if selected != st.session_state["collection"]:
            st.session_state["collection"] = selected
            st.session_state["confirm_delete"] = False
            st.rerun()

        # Create new collection
        with st.expander("Create new collection"):
            new_col = st.text_input(
                "Collection name",
                placeholder="e.g. legal-contracts, hr-policies, research-papers",
                key="_new_col_input",
            )
            if st.button("Create", key="_create_col") and new_col.strip():
                name = new_col.strip().lower().replace(" ", "-")
                st.session_state["collection"] = name
                st.session_state["confirm_delete"] = False
                st.success(f"Collection '{name}' will be created on first upload.")
                st.rerun()

        # Clear collection (remove all documents, keep the collection)
        with st.expander("Clear collection", expanded=False):
            st.warning(f"Remove all documents from **{st.session_state['collection']}**? The collection itself will be kept.")
            confirmed_clear = st.checkbox("I understand, clear all documents", key="_clear_confirm")
            if st.button("Clear", key="_clear_col", disabled=not confirmed_clear):
                try:
                    rag.clear_data(st.session_state["collection"])
                    st.success(f"All documents removed from **{st.session_state['collection']}**.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Clear failed: {exc}")

        # Delete collection
        with st.expander("Delete collection", expanded=False):
            st.warning(f"Delete **{st.session_state['collection']}**? This cannot be undone.")
            confirmed = st.checkbox("I understand, delete it", key="_del_confirm")
            if st.button("Delete", key="_del_col", disabled=not confirmed):
                try:
                    rag.delete_collection(st.session_state["collection"])
                    remaining = rag.list_collections()
                    st.session_state["collection"] = (
                        remaining[0] if remaining else Config.COLLECTION_NAME
                    )
                    st.session_state["confirm_delete"] = False
                    st.success("Collection deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

        st.divider()

        # ---- LLM Settings ----
        with st.expander("LLM Settings", expanded=False):
            provider = st.selectbox(
                "Provider",
                options=["ollama", "anthropic", "openai"],
                index=["ollama", "anthropic", "openai"].index(
                    st.session_state["llm_provider"]
                ),
                key="_llm_provider",
            )
            if provider != st.session_state["llm_provider"]:
                st.session_state["llm_provider"] = provider
                st.session_state["llm_model"] = _PROVIDER_DEFAULTS[provider]

            st.session_state["llm_model"] = st.text_input(
                "Model",
                value=st.session_state["llm_model"],
                key="_llm_model_input",
            )

            if provider != "ollama":
                st.session_state["llm_api_key"] = st.text_input(
                    "API Key",
                    value=st.session_state["llm_api_key"],
                    type="password",
                    placeholder="Leave blank to use key from .env",
                    key="_llm_api_key_input",
                )

            st.session_state["temperature"] = st.slider(
                "Temperature",
                min_value=0.0, max_value=1.0,
                value=st.session_state["temperature"], step=0.05,
            )
            st.caption("Lower = more precise/factual. Higher = more creative/varied.")

        # ---- System Prompt ----
        with st.expander("System Prompt", expanded=False):
            st.caption("Customise how the assistant responds. Changes take effect on the next query.")
            st.session_state["system_prompt"] = st.text_area(
                "Prompt",
                value=st.session_state["system_prompt"],
                height=140,
                label_visibility="collapsed",
                key="_system_prompt_input",
            )
            if st.button("Reset to default", key="_reset_prompt"):
                st.session_state["system_prompt"] = Config.SYSTEM_PROMPT
                st.rerun()

        # ---- Chunking Settings ----
        with st.expander("Chunking Settings", expanded=False):
            st.session_state["chunk_size"] = st.slider(
                "Chunk size (chars)",
                min_value=128, max_value=2048,
                value=st.session_state["chunk_size"], step=64,
                help="Larger = more context per result. Smaller = more precise retrieval.",
            )
            st.session_state["chunk_overlap"] = st.slider(
                "Overlap (chars)",
                min_value=0, max_value=200,
                value=st.session_state["chunk_overlap"], step=10,
                help="Characters shared between consecutive chunks to avoid splitting answers at boundaries.",
            )

            _strategy_labels = [STRATEGY_INFO[s]["label"] for s in STRATEGIES]
            _current_strategy = st.session_state["chunk_strategy"]
            _current_index = STRATEGIES.index(_current_strategy) if _current_strategy in STRATEGIES else 0

            _selected_label = st.selectbox(
                "Strategy",
                options=_strategy_labels,
                index=_current_index,
            )
            _selected_strategy = STRATEGIES[_strategy_labels.index(_selected_label)]
            st.session_state["chunk_strategy"] = _selected_strategy
            st.info(STRATEGY_INFO[_selected_strategy]["description"])

        # ---- Retrieval Settings ----
        with st.expander("Retrieval Settings", expanded=False):
            st.session_state["top_k_default"] = st.slider(
                "Default Top K",
                min_value=1, max_value=20,
                value=st.session_state["top_k_default"], step=1,
            )
            st.caption("Number of context chunks retrieved per query. More = broader context, slower response.")

            _r_labels = [RETRIEVAL_STRATEGY_INFO[s]["label"] for s in RETRIEVAL_STRATEGIES]
            _r_current = st.session_state["retrieval_strategy"]
            _r_index = RETRIEVAL_STRATEGIES.index(_r_current) if _r_current in RETRIEVAL_STRATEGIES else 0
            _selected_r_label = st.selectbox("Retrieval strategy", _r_labels, index=_r_index)
            _selected_r = RETRIEVAL_STRATEGIES[_r_labels.index(_selected_r_label)]
            st.session_state["retrieval_strategy"] = _selected_r
            st.info(RETRIEVAL_STRATEGY_INFO[_selected_r]["description"])

            if _selected_r == "hybrid":
                st.session_state["semantic_weight"] = st.slider(
                    "Semantic weight",
                    min_value=0.1, max_value=0.9,
                    value=st.session_state["semantic_weight"], step=0.05,
                    key="_semantic_weight_slider",
                )
                kw = round(1.0 - st.session_state["semantic_weight"], 2)
                st.caption(
                    f"Semantic {st.session_state['semantic_weight']:.0%} · "
                    f"Keyword {kw:.0%}. "
                    "Higher semantic = better for conceptual queries; "
                    "higher keyword = better for exact terms/codes."
                )

            if _selected_r == "parent-child":
                st.warning(
                    "Parent-child requires documents to be re-ingested "
                    "with this strategy active to take effect."
                )

        st.divider()

        # ---- Collection stats ----
        st.subheader("Stats")
        if st.button("Refresh", key="_refresh_stats"):
            st.cache_data.clear()
        try:
            stats = rag.get_collection_stats(st.session_state["collection"])
            st.metric("Documents", stats.get("document_count", "—"))
            st.caption(f"Embedding: `{stats.get('embedding_model', '—')}`")
            st.caption(
                f"Chunk: {st.session_state['chunk_size']} chars "
                f"· overlap {st.session_state['chunk_overlap']} "
                f"· {st.session_state['chunk_strategy']}"
            )
        except Exception as exc:
            st.caption(f"Could not load stats: {exc}")


# ---------------------------------------------------------------------------
# Upload tab
# ---------------------------------------------------------------------------
def render_upload_tab(rag: RAGSystem) -> None:
    collection = st.session_state["collection"]

    st.header("Upload Documents")
    st.caption(
        f"Uploading into collection: **{collection}**  \n"
        "Supported formats: PDF, DOCX, TXT, MD, CSV, JSON"
    )

    _form_key = st.session_state["upload_form_key"]

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md", "csv", "json"],
        label_visibility="collapsed",
        key=f"_file_uploader_{_form_key}",
    )

    with st.form(f"upload_form_{_form_key}"):
        col1, col2 = st.columns(2)

        default_title = Path(uploaded_file.name).stem if uploaded_file else ""
        default_source = uploaded_file.name if uploaded_file else ""

        with col1:
            title = st.text_input(
                "Title *",
                value=default_title,
                placeholder="e.g. Q4 Financial Report 2025",
            )
            category = st.text_input(
                "Category *",
                placeholder="e.g. Finance, HR, Legal, Technical",
            )
            source = st.text_input(
                "Source",
                value=default_source,
                placeholder="e.g. Internal Audit Team, World Bank",
            )

        with col2:
            author = st.text_input(
                "Author",
                placeholder="e.g. Jane Smith",
            )
            tags = st.text_input(
                "Tags",
                placeholder="e.g. annual-report, budget, 2025",
            )
            description = st.text_area(
                "Description",
                placeholder="Brief summary of what this document contains and why it's being added.",
                height=122,
            )

        submitted = st.form_submit_button(
            "Ingest Document", type="primary", use_container_width=True
        )

    if submitted:
        _handle_upload(
            rag=rag,
            uploaded_file=uploaded_file,
            title=title,
            category=category,
            source=source or default_source,
            description=description,
            tags=tags,
            author=author,
            collection_name=collection,
            chunk_size=st.session_state["chunk_size"],
            chunk_overlap=st.session_state["chunk_overlap"],
            chunk_strategy=st.session_state["chunk_strategy"],
        )


def _handle_upload(
    rag: RAGSystem,
    uploaded_file,
    title: str,
    category: str,
    source: str,
    description: str,
    tags: str,
    author: str,
    collection_name: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    chunk_strategy: str = None,
) -> None:
    if not uploaded_file:
        st.error("Please select a file before submitting.")
        return
    if not title.strip():
        st.error("Title is required.")
        return
    if not category.strip():
        st.error("Category is required.")
        return

    file_bytes = uploaded_file.read()
    suffix = Path(uploaded_file.name).suffix.lower()

    with st.spinner(f"Processing '{uploaded_file.name}'…"):
        try:
            if suffix in {".csv", ".json"}:
                source_type = "csv" if suffix == ".csv" else "json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                try:
                    num_chunks = rag.ingest_data(
                        source=tmp_path,
                        source_type=source_type,
                        collection_name=collection_name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        chunk_strategy=chunk_strategy,
                    )
                finally:
                    os.unlink(tmp_path)
            else:
                text = DocumentParser.parse(file_bytes, uploaded_file.name)
                num_chunks = rag.ingest_document_with_metadata(
                    text=text,
                    title=title,
                    category=category,
                    source=source,
                    description=description,
                    tags=tags,
                    author=author,
                    collection_name=collection_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_strategy=chunk_strategy,
                )

            st.success(
                f"**{uploaded_file.name}** ingested into **{collection_name}** — "
                f"**{num_chunks} chunks** added."
            )
            st.session_state["upload_form_key"] += 1
            st.rerun()

        except ValueError as exc:
            st.error(f"Could not process file: {exc}")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
def _render_sources(sources: list) -> None:
    """Render a collapsible sources expander."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            label = meta.get("title") or meta.get("source", "unknown")
            cat = meta.get("category", "—")
            score = src.get("similarity_score")
            score_str = f"{score:.3f}" if isinstance(score, float) else "—"
            st.markdown(
                f"**{i}.** `{label}` &nbsp;|&nbsp; "
                f"category: `{cat}` &nbsp;|&nbsp; "
                f"similarity: `{score_str}`"
            )
            st.caption(src.get("text", "")[:300])
            if i < len(sources):
                st.divider()


def render_chat_tab(rag: RAGSystem) -> None:
    collection = st.session_state["collection"]

    if collection not in st.session_state["chat_history"]:
        st.session_state["chat_history"][collection] = []

    history = st.session_state["chat_history"][collection]
    turns = len(history) // 2

    col_header, col_clear = st.columns([6, 1])
    with col_header:
        st.header("Chat")
        st.caption(
            f"Collection: **{collection}** · "
            f"LLM: **{st.session_state['llm_provider']} / {st.session_state['llm_model']}** · "
            f"{turns} turn{'s' if turns != 1 else ''}"
        )
    with col_clear:
        st.write("")  # vertical alignment nudge
        if history and st.button("Clear", key="_clear_chat"):
            st.session_state["chat_history"][collection] = []
            st.rerun()

    # Render conversation history
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                ms = msg.get("perf", {}).get("query_processing", {}).get("duration_ms")
                if ms:
                    st.caption(f"{ms:.0f} ms")

    # Sticky chat input
    if question := st.chat_input("Ask a question about your documents…"):
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Pass all prior turns as context (exclude the message we just added)
        llm_history = history[:-1]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = rag.query(
                        question,
                        top_k=st.session_state["top_k_default"],
                        collection_name=collection,
                        provider=st.session_state["llm_provider"],
                        model=st.session_state["llm_model"],
                        api_key=st.session_state.get("llm_api_key") or None,
                        temperature=st.session_state["temperature"],
                        system_prompt=st.session_state["system_prompt"],
                        retrieval_strategy=st.session_state["retrieval_strategy"],
                        semantic_weight=st.session_state["semantic_weight"],
                        conversation_history=llm_history,
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    history.pop()
                    st.stop()

            answer = response.get("answer_with_citations", response.get("answer", ""))
            sources = response.get("sources", [])
            perf = response.get("performance", {})

            st.markdown(answer)
            if sources:
                _render_sources(sources)
            ms = perf.get("query_processing", {}).get("duration_ms")
            if ms:
                st.caption(f"{ms:.0f} ms")

        history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "perf": perf,
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="RAG Document Manager",
        page_icon="📄",
        layout="wide",
    )

    rag = get_rag_system()
    _init_state(rag)

    render_sidebar(rag)

    tab_upload, tab_chat = st.tabs(["Upload Documents", "Chat"])

    with tab_upload:
        render_upload_tab(rag)

    with tab_chat:
        render_chat_tab(rag)


if __name__ == "__main__":
    main()
