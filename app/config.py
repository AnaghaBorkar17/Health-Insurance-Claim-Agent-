"""Application Configuration Module.

Loads configuration settings from environment variables and .env file
using Pydantic Settings. Provides centralized access to model names,
database paths, and service URLs.
"""

import os
from pathlib import Path
from typing import Optional

# Automatically bridge Streamlit Community Cloud secrets into os.environ if running inside Streamlit
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for _sec_key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "OLLAMA_BASE_URL", "OLLAMA_LLM_MODEL", "OLLAMA_EMBED_MODEL"]:
            if _sec_key in st.secrets and not os.environ.get(_sec_key):
                os.environ[_sec_key] = str(st.secrets[_sec_key])
except Exception:
    pass

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

_OLLAMA_STATUS_CACHE: dict = {}


def is_ollama_available(base_url: Optional[str] = None) -> bool:
    """Quickly check whether local Ollama server is running and reachable."""
    import time
    import requests

    url = (base_url or settings.ollama_base_url).rstrip("/")
    now = time.time()
    if url in _OLLAMA_STATUS_CACHE:
        status, ts = _OLLAMA_STATUS_CACHE[url]
        if now - ts < 30.0:
            return status

    try:
        resp = requests.get(f"{url}/api/tags", timeout=0.8)
        is_up = resp.status_code == 200
    except Exception:
        is_up = False

    _OLLAMA_STATUS_CACHE[url] = (is_up, now)
    return is_up

