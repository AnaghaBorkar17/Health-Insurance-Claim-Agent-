"""Application Configuration Module.

Loads configuration settings from environment variables and .env file
using Pydantic Settings. Provides centralized access to model names,
database paths, and service URLs.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory for the claim-agents project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with validation and defaults."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google Gemini API (Cloud LLM)
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description="Google Gemini API Key from Google AI Studio",
    )
    gemini_model: str = Field(
        default="gemini-3.8-flash",
        validation_alias="GEMINI_MODEL",
        description="Configurable Gemini model identifier",
    )

    # Ollama (Local LLM & Embeddings)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
        description="Base URL for local Ollama instance",
    )
    ollama_llm_model: str = Field(
        default="qwen2.5:3b",
        validation_alias="OLLAMA_LLM_MODEL",
        description="Local Ollama LLM for offline extraction/tasks",
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        validation_alias="OLLAMA_EMBED_MODEL",
        description="Local Ollama embedding model for ChromaDB",
    )

    # Vector Storage (ChromaDB)
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "chroma"),
        validation_alias="CHROMA_PERSIST_DIR",
        description="Filesystem path for persistent vector database",
    )

    # Relational Database & Checkpointing (SQLite)
    database_path: str = Field(
        default=str(PROJECT_ROOT / "data" / "claims.db"),
        validation_alias="DATABASE_PATH",
        description="Filesystem path for SQLite database",
    )

    # Application Environment
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging verbosity level",
    )
    app_env: str = Field(
        default="development",
        validation_alias="APP_ENV",
        description="Current runtime environment (development, test, production)",
    )

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a genuine Gemini API key has been provided."""
        if not self.gemini_api_key:
            return False
        placeholder_values = {
            "your_gemini_api_key_here",
            "your_api_key",
            "replace_me",
            "",
        }
        return self.gemini_api_key.strip().lower() not in placeholder_values


# Singleton instance for access throughout the application
settings = Settings()
