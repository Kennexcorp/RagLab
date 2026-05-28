"""
RAG Lab — Streamlit GUI
Manage collections, upload documents, and query your knowledge base.

Run with:
    streamlit run gui_app.py
"""

from __future__ import annotations

import os
import sys
import time
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

# Alternating background colours for the chunking visualizer
_CHUNK_COLORS = ["#FFF3CD", "#D1ECF1", "#D4EDDA", "#F8D7DA", "#E2D9F3", "#FDEBD0"]


# ---------------------------------------------------------------------------
# .env persistence helper
# ---------------------------------------------------------------------------
def _save_to_env(updates: dict) -> None:
    """Update KEY=VALUE lines in .env in-place, preserving all comments and structure."""
    env_path = Path(__file__).parent / ".env"
    lines = env_path.read_text().splitlines(keepends=True)
    remaining = dict(updates)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                new_lines.append(f"{key}={remaining.pop(key)}\n")
                continue
        new_lines.append(line)
    for key, val in remaining.items():
        new_lines.append(f"{key}={val}\n")
    env_path.write_text("".join(new_lines))


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
    defaults = {
        "collection": Config.COLLECTION_NAME,
        "llm_provider": Config.LLM_PROVIDER,
        "llm_model": Config.LLM_MODEL,
        "llm_api_key": "",
        "confirm_delete": False,
        "upload_form_key": 0,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "chunk_strategy": Config.CHUNKING_STRATEGY,
        "temperature": Config.LLM_TEMPERATURE,
        "system_prompt": Config.SYSTEM_PROMPT,
        "top_k_default": Config.TOP_K_RESULTS,
        "retrieval_strategy": Config.RETRIEVAL_STRATEGY,
        "semantic_weight": float(Config.SEMANTIC_WEIGHT),
        # Query tab
        "last_query_result": {},  # {collection: result_dict}
        # Educational features
        "show_pipeline_trace": False,
        "chunking_preview_text": "",
        "compare_results": {},
        # Chunk tab — independent preview controls (don't affect ingestion)
        "chunk_tab_size": Config.CHUNK_SIZE,
        "chunk_tab_overlap": Config.CHUNK_OVERLAP,
        "chunk_tab_strategy": Config.CHUNKING_STRATEGY,
        # Retrieval tab — independent controls (don't affect chat until Set as Default)
        "ret_tab_top_k": Config.TOP_K_RESULTS,
        "ret_tab_semantic_weight": float(Config.SEMANTIC_WEIGHT),
        "ret_tab_similarity_threshold": float(Config.SIMILARITY_THRESHOLD),
        "ret_tab_strategy": Config.RETRIEVAL_STRATEGY,
        # Explore tab — UMAP projection parameters
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(rag: RAGSystem) -> None:
    with st.sidebar:
        st.title("RAG Lab")

        # ---- Collections section ----
        st.subheader("Collections")

        _all = rag.list_collections()
        _default = Config.COLLECTION_NAME
        _others = sorted(c for c in _all if c != _default)
        collections = [_default] + _others if _default in _all else ([_default] + _others or [_default])

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

        with st.expander("Create new collection"):
            new_col = st.text_input(
                "Collection name",
                placeholder="e.g. legal-contracts, hr-policies",
                key="_new_col_input",
            )
            if st.button("Create", key="_create_col") and new_col.strip():
                name = new_col.strip().lower().replace(" ", "-")
                st.session_state["collection"] = name
                st.session_state["confirm_delete"] = False
                st.success(f"Collection '{name}' will be created on first upload.")
                st.rerun()

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
                index=["ollama", "anthropic", "openai"].index(st.session_state["llm_provider"]),
                key="_llm_provider",
            )
            if provider != st.session_state["llm_provider"]:
                st.session_state["llm_provider"] = provider
                st.session_state["llm_model"] = _PROVIDER_DEFAULTS[provider]

            st.session_state["llm_model"] = st.text_input(
                "Model", value=st.session_state["llm_model"], key="_llm_model_input"
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

        # ---- Active retrieval settings (read-only — configured in Retrieval tab) ----
        _active_strategy = st.session_state["retrieval_strategy"]
        _active_label = RETRIEVAL_STRATEGY_INFO.get(_active_strategy, {}).get("label", _active_strategy)
        st.caption(
            f"**Retrieval:** {_active_label}  \n"
            f"Top K: {st.session_state['top_k_default']} · "
            f"Threshold: {st.session_state.get('ret_tab_similarity_threshold', Config.SIMILARITY_THRESHOLD):.2f}  \n"
            f"*Configure in the Retrieval tab.*"
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

        st.divider()

        # ---- Educational Tools ----
        st.subheader("Educational Tools")
        st.session_state["show_pipeline_trace"] = st.toggle(
            "Show pipeline trace",
            value=st.session_state.get("show_pipeline_trace", False),
            key="_show_trace_toggle",
            help=(
                "After each chat response, show a collapsible trace of the full RAG pipeline: "
                "query rewriting, retrieved chunks, the exact prompt sent to the LLM, and generation stats."
            ),
        )


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
            title = st.text_input("Title *", value=default_title, placeholder="e.g. Q4 Financial Report 2025")
            category = st.text_input("Category *", placeholder="e.g. Finance, HR, Legal, Technical")
            source = st.text_input("Source", value=default_source, placeholder="e.g. Internal Audit Team")

        with col2:
            author = st.text_input("Author", placeholder="e.g. Jane Smith")
            tags = st.text_input("Tags", placeholder="e.g. annual-report, budget, 2025")
            description = st.text_area(
                "Description",
                placeholder="Brief summary of what this document contains.",
                height=122,
            )

        submitted = st.form_submit_button("Ingest Document", type="primary", use_container_width=True)

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
                # Pre-populate chunking visualizer with this document's text
                st.session_state["chunking_preview_text"] = text
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
# Feature 3 — Chunking Visualizer
# ---------------------------------------------------------------------------
def render_chunk_tab() -> None:
    """Standalone chunking visualizer tab with its own controls."""
    st.header("Chunking Visualizer")
    st.caption(
        "Experiment with chunking settings and see exactly how your text will be split — "
        "nothing is stored. These controls are independent of the ingestion settings in the sidebar. "
        "After uploading a document, the text area below auto-populates with that document's text."
    )

    # ---- Controls ----
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        chunk_size = st.slider(
            "Chunk size (chars)",
            min_value=100, max_value=4000,
            value=st.session_state["chunk_tab_size"],
            step=50,
            key="_ctab_size",
        )
        st.session_state["chunk_tab_size"] = chunk_size
    with col_b:
        chunk_overlap = st.slider(
            "Overlap (chars)",
            min_value=0, max_value=500,
            value=st.session_state["chunk_tab_overlap"],
            step=10,
            key="_ctab_overlap",
        )
        st.session_state["chunk_tab_overlap"] = chunk_overlap
    with col_c:
        strategy = st.selectbox(
            "Strategy",
            options=STRATEGIES,
            index=STRATEGIES.index(st.session_state["chunk_tab_strategy"]),
            format_func=lambda s: STRATEGY_INFO[s]["label"],
            key="_ctab_strategy",
        )
        st.session_state["chunk_tab_strategy"] = strategy
        st.caption(STRATEGY_INFO[strategy]["description"])

    st.divider()

    # ---- Text input ----
    preview_text = st.text_area(
        "Text to preview",
        value=st.session_state.get("chunking_preview_text", ""),
        height=180,
        placeholder="Paste any text here, or upload a document to auto-populate…",
        key="_chunk_preview_input",
    )
    st.session_state["chunking_preview_text"] = preview_text

    col_btn, col_default = st.columns([3, 1])
    run_preview = col_btn.button("Preview Chunks", type="primary", key="_preview_chunks", use_container_width=True)
    if col_default.button("Set as Default", key="_chunk_set_default", use_container_width=True):
        st.session_state["chunk_size"] = chunk_size
        st.session_state["chunk_overlap"] = chunk_overlap
        st.session_state["chunk_strategy"] = strategy
        _save_to_env({
            "CHUNK_SIZE": chunk_size,
            "CHUNK_OVERLAP": chunk_overlap,
            "CHUNKING_STRATEGY": strategy,
        })
        st.success(f"Defaults saved — size {chunk_size}, overlap {chunk_overlap}, strategy {strategy}.")

    if run_preview:
        if not preview_text.strip():
            st.warning("Paste some text to preview.")
            return
        try:
            from chunking import TextChunker
            from utils import count_tokens

            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy,
            )
            doc = {"text": preview_text, "metadata": {"source": "preview"}}
            chunks = chunker.chunk_document(doc)

            if strategy == "semantic-embedding":
                st.info(
                    "Semantic Embedding splits on meaning changes, not character count — "
                    "chunk size and overlap settings are not used.",
                    icon="ℹ️",
                )
            st.success(f"{len(chunks)} chunk{'s' if len(chunks) != 1 else ''} produced.")

            html_parts = []
            for i, chunk in enumerate(chunks):
                color = _CHUNK_COLORS[i % len(_CHUNK_COLORS)]
                body = chunk["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                char_count = len(chunk["text"])
                token_count = count_tokens(chunk["text"])
                badge = (
                    f'<span style="font-size:0.74em;background:#444;color:#fff;'
                    f'border-radius:4px;padding:2px 7px;display:inline-block;margin-bottom:5px;">'
                    f'Chunk {i + 1} &nbsp;·&nbsp; {char_count} chars &nbsp;·&nbsp; ~{token_count} tokens'
                    f'</span>'
                )
                block = (
                    f'<div style="background:{color};border-radius:6px;'
                    f'padding:10px 14px;margin-bottom:8px;color:#1a1a1a;">'
                    f'{badge}<br/>'
                    f'<pre style="white-space:pre-wrap;font-size:0.82em;margin:5px 0 0 0;'
                    f'font-family:monospace;color:#1a1a1a;">{body}</pre>'
                    f'</div>'
                )
                html_parts.append(block)

            st.markdown("".join(html_parts), unsafe_allow_html=True)

        except Exception as exc:
            st.error(f"Chunking preview failed: {exc}")


# ---------------------------------------------------------------------------
# Feature 5 — Enhanced Sources Expander
# ---------------------------------------------------------------------------
def _render_sources(
    sources: list,
    retrieval_ms: float = None,
    context_tokens: int = None,
) -> None:
    """Render a collapsible sources expander with score gauges and metadata."""
    strategy = st.session_state.get("retrieval_strategy", "semantic")
    if strategy == "reranking":
        score_label = "cross-encoder score"
    elif strategy == "semantic":
        score_label = "cosine similarity"
    else:
        score_label = "rank-based score"

    header = f"Sources ({len(sources)})"
    if retrieval_ms:
        header += f"  ·  {retrieval_ms:.0f} ms"

    with st.expander(header):
        if context_tokens:
            st.caption(f"Context: ~{context_tokens:,} tokens fed to the LLM")
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            label = meta.get("title") or meta.get("source", "unknown")
            cat = meta.get("category", "—")
            score = src.get("similarity_score")
            score_val = score if isinstance(score, float) else 0.0
            score_str = f"{score_val:.3f}" if isinstance(score, float) else "—"

            st.markdown(
                f"**{i}.** `{label}` &nbsp;|&nbsp; "
                f"category: `{cat}` &nbsp;|&nbsp; "
                f"{score_label}: `{score_str}`"
            )
            st.progress(min(1.0, max(0.0, score_val)), text=score_label)
            st.caption(src.get("text", "")[:300])
            if i < len(sources):
                st.divider()


# ---------------------------------------------------------------------------
# Feature 1 + 4 — Pipeline Trace Panel
# ---------------------------------------------------------------------------
def _render_pipeline_trace(trace: dict) -> None:
    """Render the collapsible RAG pipeline trace for one assistant turn."""
    with st.expander("🔍 Pipeline Trace", expanded=False):

        # Stage 1: Query
        st.markdown("**Stage 1 · Query**")
        orig = trace.get("original_question", "")
        srch = trace.get("search_question", orig)
        if srch and srch != orig:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original question")
                st.info(orig)
            with c2:
                st.caption("Rewritten for retrieval")
                st.success(srch)
        else:
            st.caption("Question (no rewrite applied)")
            st.info(orig)

        st.divider()

        # Stage 2: Retrieval
        st.markdown("**Stage 2 · Retrieval**")
        strategy = trace.get("strategy", "—")
        ret_ms = trace.get("retrieval_ms", 0)
        st.caption(f"Strategy: `{strategy}` · Time: `{ret_ms:.0f} ms`")

        sources = trace.get("sources", [])
        if sources:
            for i, src in enumerate(sources, 1):
                meta = src.get("metadata", {})
                label = meta.get("title") or meta.get("source", "unknown")
                score = src.get("similarity_score")
                score_str = f"{score:.3f}" if isinstance(score, float) else "—"
                preview = src.get("text", "")[:200]
                st.markdown(
                    f"**Chunk {i}** &nbsp;·&nbsp; `{label}` &nbsp;·&nbsp; "
                    f"score `{score_str}`  \n_{preview}_"
                )
        else:
            st.caption("No chunks retrieved.")

        st.divider()

        # Stage 3: Prompt (Feature 4)
        st.markdown("**Stage 3 · Prompt sent to LLM**")
        prompt_sent = trace.get("prompt_sent", "")
        if prompt_sent:
            st.code(prompt_sent, language="text")
        else:
            st.caption("Prompt not captured for this turn.")

        st.divider()

        # Stage 4: Generation
        st.markdown("**Stage 4 · Generation**")
        gen_ms = trace.get("generation_ms", 0)
        total_ms = trace.get("total_ms", 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", trace.get("model", "—"))
        c2.metric("Tokens", trace.get("tokens_used") or "—")
        c3.metric("Finish", trace.get("finish_reason") or "—")
        c4.metric("Gen time", f"{gen_ms:.0f} ms")
        st.caption(f"Total pipeline: {total_ms:.0f} ms")


# ---------------------------------------------------------------------------
# Query tab
# ---------------------------------------------------------------------------
def render_chat_tab(rag: RAGSystem) -> None:
    collection = st.session_state["collection"]

    _ret_strat = st.session_state["retrieval_strategy"]
    _ret_label = RETRIEVAL_STRATEGY_INFO.get(_ret_strat, {}).get("label", _ret_strat)
    _chunk_strat = st.session_state["chunk_strategy"]
    _chunk_label = STRATEGY_INFO.get(_chunk_strat, {}).get("label", _chunk_strat)

    st.header("Query")
    st.caption(
        f"Collection: **{collection}** · "
        f"LLM: **{st.session_state['llm_provider']} / {st.session_state['llm_model']}**  \n"
        f"Retrieval: **{_ret_label}** · "
        f"Top K: **{st.session_state['top_k_default']}** · "
        f"Threshold: **{st.session_state.get('ret_tab_similarity_threshold', Config.SIMILARITY_THRESHOLD):.2f}**  \n"
        f"Chunking: **{_chunk_label}** · "
        f"Size: **{st.session_state['chunk_size']}** · "
        f"Overlap: **{st.session_state['chunk_overlap']}**"
    )

    st.divider()

    col_input, col_btn = st.columns([5, 1])
    question = col_input.text_input(
        "Query",
        placeholder="Ask a question about your documents…",
        label_visibility="collapsed",
        key="_query_input",
    )
    ask = col_btn.button("Ask", type="primary", use_container_width=True, key="_ask_btn")

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

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
                )
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.stop()

        perf = response.get("performance", {})
        total_ms = perf.get("query_processing", {}).get("duration", 0) * 1000
        trace = None
        if st.session_state.get("show_pipeline_trace"):
            trace = {
                "original_question": question,
                "search_question": response.get("search_question", question),
                "strategy": _ret_strat,
                "retrieval_ms": perf.get("retrieval", {}).get("duration", 0) * 1000,
                "generation_ms": perf.get("generation", {}).get("duration", 0) * 1000,
                "total_ms": total_ms,
                "sources": response.get("sources", []),
                "prompt_sent": response.get("prompt_sent", ""),
                "model": response.get("model", "—"),
                "tokens_used": response.get("tokens_used"),
                "finish_reason": response.get("finish_reason"),
            }

        st.session_state["last_query_result"][collection] = {
            "question": question,
            "answer": response.get("answer_with_citations", response.get("answer", "")),
            "sources": response.get("sources", []),
            "perf": perf,
            "context_tokens": response.get("context_tokens"),
            "total_ms": total_ms,
            "trace": trace,
        }

    result = st.session_state["last_query_result"].get(collection)
    if not result:
        st.info("Enter a question above to query your collection.")
        return

    st.divider()
    st.markdown(f"**Q: {result['question']}**")
    st.markdown(result["answer"])

    sources = result.get("sources", [])
    perf = result.get("perf", {})
    if sources:
        ret_ms = perf.get("retrieval", {}).get("duration", 0) * 1000
        _render_sources(sources, retrieval_ms=ret_ms, context_tokens=result.get("context_tokens"))

    if result.get("total_ms"):
        st.caption(f"{result['total_ms']:.0f} ms total")

    if st.session_state.get("show_pipeline_trace") and result.get("trace"):
        _render_pipeline_trace(result["trace"])


# ---------------------------------------------------------------------------
# Feature 2 — Strategy Comparison Tab
# ---------------------------------------------------------------------------
def _build_generator_for_comparison():
    """Build a Generator using the current session LLM settings."""
    from generator import Generator
    return Generator(
        provider=st.session_state["llm_provider"],
        model=st.session_state["llm_model"],
        api_key=st.session_state.get("llm_api_key") or None,
        temperature=st.session_state["temperature"],
        system_prompt=st.session_state["system_prompt"],
        log_level="WARNING",
    )


def _render_compare_card(col, strategy: str, result: dict) -> None:
    info = RETRIEVAL_STRATEGY_INFO.get(strategy, {})
    label = info.get("label", strategy)
    desc = info.get("description", "")
    elapsed = result.get("elapsed_ms", 0)
    desc_short = desc[:110] + "…" if len(desc) > 110 else desc

    with col:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(f"{elapsed:.0f} ms · {desc}")

            error = result.get("error")
            if error:
                st.error(f"Error: {error}")
                return

            chunks = result.get("chunks", [])
            if not chunks:
                st.warning("No results returned.")
            else:
                for i, chunk in enumerate(chunks, 1):
                    score = chunk.get("similarity_score")
                    score_str = f"{score:.3f}" if isinstance(score, float) else "—"
                    meta = chunk.get("metadata", {})
                    src = meta.get("title") or meta.get("source", "unknown")
                    preview = chunk.get("text", "")[:110]
                    st.markdown(
                        f"**#{i}** `{src}` · score `{score_str}`  \n_{preview}_"
                    )

            if result.get("llm_answer"):
                st.divider()
                st.markdown("**LLM Answer**")
                st.markdown(result["llm_answer"])


def render_compare_tab(rag: RAGSystem) -> None:
    collection = st.session_state["collection"]

    st.header("Retrieval")
    st.caption(
        "Configure your retrieval strategy and parameters, then run a comparison across all 8 strategies "
        "to validate your choices. Use **Set as Default** to apply settings to Chat."
    )
    st.info(
        "Multi-Query, HyDE, and Self-Query use the LLM configured in `.env` for retrieval-time "
        "generation, not the session LLM settings.",
        icon="ℹ️",
    )

    # ---- Strategy selector ----
    ret_strategy = st.selectbox(
        "Chat strategy",
        options=RETRIEVAL_STRATEGIES,
        index=RETRIEVAL_STRATEGIES.index(st.session_state["ret_tab_strategy"])
              if st.session_state["ret_tab_strategy"] in RETRIEVAL_STRATEGIES else 0,
        format_func=lambda s: RETRIEVAL_STRATEGY_INFO[s]["label"],
        key="_ret_strategy_select",
        help="The strategy Chat will use when answering questions.",
    )
    st.session_state["ret_tab_strategy"] = ret_strategy
    st.caption(RETRIEVAL_STRATEGY_INFO[ret_strategy]["description"])
    if ret_strategy == "parent-child":
        st.warning("Parent-child requires documents to be re-ingested with this strategy active.")

    st.divider()

    # ---- Parameter controls ----
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ret_top_k = st.slider(
            "Top K results",
            min_value=1, max_value=20,
            value=st.session_state["ret_tab_top_k"],
            step=1,
            key="_ret_top_k",
            help="Number of chunks each strategy retrieves.",
        )
        st.session_state["ret_tab_top_k"] = ret_top_k
    with col_b:
        ret_semantic_weight = st.slider(
            "Semantic weight (hybrid)",
            min_value=0.0, max_value=1.0,
            value=st.session_state["ret_tab_semantic_weight"],
            step=0.05,
            key="_ret_sem_weight",
            help="Hybrid strategy only — balance between semantic (1.0) and BM25 keyword (0.0).",
        )
        st.session_state["ret_tab_semantic_weight"] = ret_semantic_weight
    with col_c:
        ret_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0, max_value=1.0,
            value=st.session_state["ret_tab_similarity_threshold"],
            step=0.05,
            key="_ret_threshold",
            help="Chunks scoring below this are excluded. Lower = more results.",
        )
        st.session_state["ret_tab_similarity_threshold"] = ret_threshold

    col_run, col_default = st.columns([3, 1])
    if col_default.button("Set as Default", key="_ret_set_default", use_container_width=True):
        st.session_state["top_k_default"] = ret_top_k
        st.session_state["semantic_weight"] = ret_semantic_weight
        st.session_state["retrieval_strategy"] = ret_strategy
        _, _ret = rag._get_or_create_collection(collection)
        _ret.similarity_threshold = ret_threshold
        _save_to_env({
            "TOP_K_RESULTS": ret_top_k,
            "SEMANTIC_WEIGHT": ret_semantic_weight,
            "KEYWORD_WEIGHT": round(1.0 - ret_semantic_weight, 4),
            "SIMILARITY_THRESHOLD": ret_threshold,
            "RETRIEVAL_STRATEGY": ret_strategy,
        })
        col_default.success("Saved.")

    st.divider()

    query = st.text_input(
        "Query",
        value=st.session_state.get("compare_query", ""),
        placeholder="e.g. What are the key principles described in this document?",
        key="_compare_query_input",
    )
    include_llm = st.checkbox(
        "Include LLM answers (slower — one generation per strategy)",
        value=False,
        key="_compare_include_llm",
    )

    if col_run.button("Run Comparison", type="primary", key="_run_compare", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
            st.stop()

        st.session_state["compare_query"] = query
        # Capture all values before the loop to avoid "dictionary changed size
        # during iteration" from st.session_state reads inside the loop.
        semantic_weight = ret_semantic_weight
        top_k = ret_top_k
        similarity_threshold = ret_threshold
        results = {}
        _, ret = rag._get_or_create_collection(collection)
        ret.similarity_threshold = similarity_threshold

        status = st.empty()
        for idx, strategy in enumerate(RETRIEVAL_STRATEGIES):
            status.caption(f"Running {strategy} ({idx + 1}/{len(RETRIEVAL_STRATEGIES)})…")
            t0 = time.time()
            try:
                chunks = ret.retrieve(
                    query,
                    top_k=top_k,
                    strategy=strategy,
                    semantic_weight=semantic_weight,
                )
                elapsed_ms = (time.time() - t0) * 1000

                llm_answer = None
                if include_llm and chunks:
                    try:
                        from utils import format_context_for_llm
                        ctx = format_context_for_llm(chunks)
                        gen = _build_generator_for_comparison()
                        gen_resp = gen.generate(query, ctx)
                        llm_answer = gen_resp.get("answer", "")
                    except Exception as gen_exc:
                        llm_answer = f"_(Generation failed: {gen_exc})_"

                results[strategy] = {
                    "chunks": chunks,
                    "elapsed_ms": elapsed_ms,
                    "error": None,
                    "llm_answer": llm_answer,
                }
            except Exception as exc:
                results[strategy] = {
                    "chunks": [],
                    "elapsed_ms": (time.time() - t0) * 1000,
                    "error": str(exc),
                    "llm_answer": None,
                }

        status.empty()
        st.session_state["compare_results"] = results

    results = st.session_state.get("compare_results", {})
    if not results:
        st.info("Enter a query and click **Run Comparison** to see results.")
        return

    strategy_list = list(results.keys())
    for row_start in range(0, len(strategy_list), 2):
        pair = strategy_list[row_start:row_start + 2]
        cols = st.columns(len(pair))
        for col, strat in zip(cols, pair):
            _render_compare_card(col, strat, results[strat])


# ---------------------------------------------------------------------------
# Feature 6 — Embedding Space Visualization
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing UMAP projection…")
def _compute_umap(
    collection_name: str,
    doc_count: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_samples: int = 500,
):
    """
    Fetch raw embeddings from ChromaDB and reduce to 2D with UMAP.
    Returns dict with xy, texts, sources, reducer — or None if umap-learn not installed.
    Cache key includes (collection_name, doc_count, n_neighbors, min_dist) so the projection
    recomputes automatically whenever any of these change.
    """
    try:
        import umap as umap_lib
        import numpy as np
    except ImportError:
        return None

    if doc_count == 0:
        return None

    rag = get_rag_system()
    vs, _ = rag._get_or_create_collection(collection_name)

    raw = vs.chroma._collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = raw.get("embeddings")
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    if embeddings is None or len(embeddings) == 0:
        return None

    emb_array = np.array(embeddings)

    if len(emb_array) > max_samples:
        idx = np.random.default_rng(42).choice(len(emb_array), max_samples, replace=False)
        emb_array = emb_array[idx]
        documents = [documents[i] for i in idx]
        metadatas = [metadatas[i] for i in idx]

    # Cap n_neighbors to avoid UMAP error on small collections
    effective_n_neighbors = min(n_neighbors, len(emb_array) - 1)
    reducer = umap_lib.UMAP(
        n_components=2, random_state=42,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
    )
    xy = reducer.fit_transform(emb_array)

    sources = [(m.get("title") or m.get("source", "unknown")) for m in metadatas]
    return {
        "xy": xy.tolist(),
        "texts": [d[:120] for d in documents],
        "sources": sources,
        "reducer": reducer,
    }


def render_explore_tab(rag: RAGSystem) -> None:
    try:
        import plotly.express as px
        import numpy as np
        import pandas as pd
    except ImportError:
        st.error("This tab requires `umap-learn` and `plotly`. Run `pip install umap-learn plotly` and restart.")
        return

    collection = st.session_state["collection"]

    st.header("Embedding Space")
    st.caption(
        "Each dot is a document chunk, positioned by semantic similarity — chunks that are "
        "conceptually close appear near each other. Colour = source document. "
        "Enter a query to see where it lands and which chunks get retrieved."
    )

    try:
        stats = rag.get_collection_stats(collection)
        doc_count = stats.get("document_count", 0)
    except Exception:
        doc_count = 0

    if doc_count == 0:
        st.info("No documents in this collection. Upload documents first.")
        return

    # ---- UMAP parameters ----
    col_a, col_b, col_c = st.columns([2, 2, 4])
    with col_a:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2, max_value=50,
            value=st.session_state["umap_n_neighbors"],
            step=1,
            key="_umap_n_neighbors",
            help=(
                "Controls how much local vs global structure UMAP preserves. "
                "Low (2–5) = tight local clusters. High (30–50) = broader global layout."
            ),
        )
        st.session_state["umap_n_neighbors"] = n_neighbors
    with col_b:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0, max_value=0.99,
            value=st.session_state["umap_min_dist"],
            step=0.05,
            key="_umap_min_dist",
            help=(
                "Minimum distance between points in the 2D projection. "
                "Low (~0.0) = tightly packed clusters. High (~0.5) = more spread out."
            ),
        )
        st.session_state["umap_min_dist"] = min_dist
    with col_c:
        st.info(
            "**Chunking → Explore connection:** The projection is built from the embeddings currently "
            "stored in ChromaDB. Changing chunk settings in the **Chunking** tab only updates the "
            "preview — to see the effect here, re-ingest your documents with the new settings, "
            "then click **Refresh Projection**.",
            icon="ℹ️",
        )

    col_btn, col_note = st.columns([2, 6])
    with col_btn:
        if st.button("Refresh Projection", key="_refresh_umap"):
            _compute_umap.clear()
            st.rerun()
    with col_note:
        st.caption(f"{doc_count:,} chunks in collection · up to 500 sampled for projection")

    umap_data = _compute_umap(collection, doc_count, n_neighbors=n_neighbors, min_dist=min_dist)
    if umap_data is None:
        st.error(
            "UMAP computation failed — ensure `umap-learn` is installed: "
            "`pip install umap-learn`"
        )
        return

    xy = np.array(umap_data["xy"])
    texts = umap_data["texts"]
    sources = umap_data["sources"]

    df = pd.DataFrame({
        "x": xy[:, 0],
        "y": xy[:, 1],
        "source": sources,
        "text": texts,
        "point_type": ["chunk"] * len(xy),
        "size": [6] * len(xy),
    })

    query = st.text_input(
        "Enter a query to project onto the map",
        placeholder="e.g. What are the revenue targets?",
        key="_explore_query",
    )

    highlight_set: set[int] = set()

    if query.strip():
        with st.spinner("Embedding query and retrieving…"):
            try:
                vs, ret = rag._get_or_create_collection(collection)
                query_emb = vs.embeddings.embed_query(query)
                reducer = umap_data["reducer"]
                query_xy = reducer.transform([query_emb])[0]

                retrieved = ret.retrieve(
                    query,
                    top_k=st.session_state["top_k_default"],
                    strategy=st.session_state["retrieval_strategy"],
                )
                retrieved_texts = {c.get("text", "")[:120] for c in retrieved}
                highlight_set = {i for i, t in enumerate(texts) if t in retrieved_texts}

                query_label = f"Query: {query[:50]}{'…' if len(query) > 50 else ''}"
                query_row = pd.DataFrame({
                    "x": [query_xy[0]], "y": [query_xy[1]],
                    "source": [query_label],
                    "text": [query],
                    "point_type": ["query"],
                    "size": [14],
                })
                df = pd.concat([df, query_row], ignore_index=True)
            except Exception as exc:
                st.warning(f"Could not project query: {exc}")

    # Build color map: query point always red, everything else auto-assigned
    query_label = f"Query: {query[:50]}{'…' if len(query) > 50 else ''}" if query.strip() else None
    color_map = {query_label: "#FF2B2B"} if query_label else {}

    fig = px.scatter(
        df, x="x", y="y",
        color="source",
        hover_data={"text": True, "x": False, "y": False, "size": False, "point_type": False},
        color_discrete_map=color_map,
        title=f"Embedding Space — {collection}",
        height=580,
    )
    fig.update_traces(marker=dict(size=6), selector=dict(mode="markers"))

    if highlight_set:
        # Dim all non-retrieved, non-query traces so retrieved chunks stand out
        for trace in fig.data:
            if trace.name != query_label and trace.name != "Retrieved":
                trace.marker.opacity = 0.2

        hi_idx = list(highlight_set)
        fig.add_scatter(
            x=xy[hi_idx, 0],
            y=xy[hi_idx, 1],
            mode="markers",
            marker=dict(size=14, color="#00C853", symbol="star", line=dict(width=1, color="#007A33"), opacity=1.0),
            name="Retrieved",
            hovertext=[texts[i] for i in hi_idx],
            hoverinfo="text",
        )

    # Make query dot larger and fully opaque
    if query_label:
        for trace in fig.data:
            if trace.name == query_label:
                trace.marker.size = 14
                trace.marker.opacity = 1.0

    fig.update_layout(
        legend_title="Source",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if query.strip() and highlight_set:
        st.caption(
            f"★ Green stars = top-{st.session_state['top_k_default']} retrieved chunks  "
            f"· Red dot = query position  "
            f"· Strategy: `{st.session_state['retrieval_strategy']}`"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="RAG Lab",
        page_icon="📄",
        layout="wide",
    )

    rag = get_rag_system()
    _init_state(rag)

    render_sidebar(rag)

    tab_upload, tab_chat, tab_compare, tab_chunk, tab_explore = st.tabs([
        "Upload Documents", "Query", "Retrieval", "Chunking", "Explore",
    ])

    with tab_upload:
        render_upload_tab(rag)

    with tab_chat:
        render_chat_tab(rag)

    with tab_compare:
        render_compare_tab(rag)

    with tab_chunk:
        render_chunk_tab()

    with tab_explore:
        render_explore_tab(rag)


if __name__ == "__main__":
    main()
