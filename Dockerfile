# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: dependencies
# The project has no [build-system], so it is not installable as a package.
# uv sync installs the locked dependencies only and the app runs from source.
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Stage 2: models
# Everything RAGLab would otherwise fetch on first use is baked in here, so the
# container starts without network access and without a cold-start download.
# ---------------------------------------------------------------------------
FROM builder AS models

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data

# Used by the "spacy" chunking strategy; not a declared dependency upstream.
RUN python -m spacy download en_core_web_sm

# Defaults of Config.EMBEDDING_MODEL and Config.RERANKER_MODEL.
RUN python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY

# Used by the "sentence" chunking strategy.
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/opt/nltk_data')"

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm AS runtime

# HF_HUB_OFFLINE is deliberately not set: EMBEDDING_MODEL and RERANKER_MODEL are
# configurable, so the image must still be able to fetch a model the user asks for.
# The defaults are cached in /opt/hf and load without downloading. For an air-gapped
# host, run with -e HF_HUB_OFFLINE=1 to skip the hub revision check entirely.
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    NLTK_DATA=/opt/nltk_data

RUN useradd --create-home --uid 10001 raglab

WORKDIR /app

COPY --from=models --chown=raglab:raglab /app/.venv /app/.venv
COPY --from=models --chown=raglab:raglab /opt/hf /opt/hf
COPY --from=models --chown=raglab:raglab /opt/nltk_data /opt/nltk_data

COPY --chown=raglab:raglab raglab/ ./raglab/
COPY --chown=raglab:raglab gui_app.py mcp_server.py ./

# Config derives DATA_DIR, VECTOR_DB_DIR and the log file from the source root and
# mkdirs them on every startup, so /app itself has to be writable by the app user.
# Mount volumes over /app/data and /app/vector_db to persist ingested collections.
RUN mkdir -p /app/data /app/vector_db && chown -R raglab:raglab /app

USER raglab

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

# Default entry point is the Streamlit GUI. Override the command for the others:
#   docker run --rm -it <image> python -m raglab.rag_system --interactive
#   docker run --rm -i  <image> python mcp_server.py
CMD ["streamlit", "run", "gui_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
