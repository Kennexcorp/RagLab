"""
Configuration management for RAG system.
Handles environment variables, model settings, and system parameters.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration for RAG system."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, ollama
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    USE_OPENAI_EMBEDDINGS = (
        os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    )

    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chromadb")  # chromadb, faiss
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dashboard_data")
    PERSIST_DIRECTORY = str(VECTOR_DB_DIR)

    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")  # semantic, fixed

    # Retrieval Configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() == "true"

    # Hybrid Search Configuration
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.7))  # 70% semantic
    KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", 0.3))  # 30% keyword
    BM25_K1 = float(os.getenv("BM25_K1", 1.5))  # BM25 term frequency saturation
    BM25_B = float(os.getenv("BM25_B", 0.75))  # BM25 length normalization
    BM25_INDEX_PATH = VECTOR_DB_DIR / "bm25_index.pkl"

    # System Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", 4000))

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings."""
        errors = []

        # Check LLM API keys based on provider
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")

        # Create necessary directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_DIR.mkdir(exist_ok=True)

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

    @classmethod
    def get_llm_api_key(cls) -> Optional[str]:
        """Get the appropriate API key based on LLM provider."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_API_KEY
        elif cls.LLM_PROVIDER == "anthropic":
            return cls.ANTHROPIC_API_KEY
        return None

    @classmethod
    def display_config(cls) -> str:
        """Display current configuration (hiding sensitive data)."""
        config_info = f"""
RAG System Configuration
========================
LLM Provider: {cls.LLM_PROVIDER}
LLM Model: {cls.LLM_MODEL}
Embedding Model: {cls.EMBEDDING_MODEL}
Vector Store: {cls.VECTOR_STORE_TYPE}
Collection: {cls.COLLECTION_NAME}
Chunk Size: {cls.CHUNK_SIZE}
Chunk Overlap: {cls.CHUNK_OVERLAP}
Top K Results: {cls.TOP_K_RESULTS}
API Key Set: {'Yes' if cls.get_llm_api_key() else 'No'}
"""
        return config_info


if __name__ == "__main__":
    # Validate and display configuration
    if Config.validate():
        print(Config.display_config())
    else:
        print("Configuration validation failed!")
