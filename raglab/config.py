"""
Configuration management for RAG system.
Handles environment variables, model settings, and system parameters.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for RAG system."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    LLM_PROVIDER: str = "ollama"  # ollama (default), openai, anthropic
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    LLM_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Vector Store Configuration
    COLLECTION_NAME: str = "documents"

    # System prompt for the LLM — override via SYSTEM_PROMPT env var
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Use the provided context to answer questions "
        "accurately and concisely. If the context doesn't contain enough information "
        "to answer the question, say so clearly."
    )

    # Chunking Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    CHUNKING_STRATEGY: str = "semantic"  # semantic, fixed

    # Retrieval Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    # Hybrid Search Configuration
    USE_HYBRID_SEARCH: bool = True
    SEMANTIC_WEIGHT: float = 0.7  # 70% semantic
    KEYWORD_WEIGHT: float = 0.3  # 30% keyword

    # Retrieval Strategy Configuration
    RETRIEVAL_STRATEGY: str = "semantic"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    PARENT_CHUNK_SIZE: int = 1024

    # System Configuration
    LOG_LEVEL: str = "INFO"
    MAX_CONTEXT_LENGTH: int = 4000

    # Project paths (derived, not env-configurable)
    @property
    def PROJECT_ROOT(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def DATA_DIR(self) -> Path:
        return self.PROJECT_ROOT / "data"

    @property
    def VECTOR_DB_DIR(self) -> Path:
        return self.PROJECT_ROOT / "vector_db"

    @property
    def PERSIST_DIRECTORY(self) -> str:
        return str(self.VECTOR_DB_DIR)

    @property
    def BM25_INDEX_PATH(self) -> Path:
        return self.VECTOR_DB_DIR / "bm25_index.pkl"

    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []

        # Check LLM API keys based on provider
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required when using Anthropic provider")

        # Create necessary directories
        self.DATA_DIR.mkdir(exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(exist_ok=True)

        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False

        return True

    def get_llm_api_key(self) -> str | None:
        """Get the appropriate API key based on LLM provider."""
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        elif self.LLM_PROVIDER == "anthropic":
            return self.ANTHROPIC_API_KEY
        return None

    def display_config(self) -> str:
        """Display current configuration (hiding sensitive data)."""
        config_info = f"""
RAG System Configuration
========================
LLM Provider: {self.LLM_PROVIDER}
LLM Model: {self.LLM_MODEL}
Embedding Model: {self.EMBEDDING_MODEL}
Collection: {self.COLLECTION_NAME}
Chunk Size: {self.CHUNK_SIZE}
Chunk Overlap: {self.CHUNK_OVERLAP}
Top K Results: {self.TOP_K_RESULTS}
API Key Set: {"Yes" if self.get_llm_api_key() else "No"}
"""
        return config_info


Config = Settings()


if __name__ == "__main__":
    # Validate and display configuration
    if Config.validate():
        print(Config.display_config())
    else:
        print("Configuration validation failed!")
