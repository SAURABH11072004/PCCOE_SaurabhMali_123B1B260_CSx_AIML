"""Centralized Configuration and Gemini Client Module.
Supports local development (.env), Streamlit Cloud (st.secrets),
multi-model configuration, safe connection verification, and zero credential leakage.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List
from dotenv import load_dotenv, dotenv_values

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_env_paths() -> List[Path]:
    """Return prioritized list of possible .env file locations."""
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent.parent / ".env",
    ]
    seen = set()
    result = []
    for c in candidates:
        try:
            resolved = c.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
        except Exception:
            pass
    return result


def reload_environment():
    """Load .env files from project hierarchy into os.environ without overwriting existing variables."""
    for p in reversed(get_env_paths()):
        if p.is_file():
            try:
                load_dotenv(p, override=False)
            except Exception:
                pass


# Load local .env into process environment if present
reload_environment()
_env_path = next((p for p in get_env_paths() if p.is_file()), PROJECT_ROOT / ".env")


def _get_secret_or_env(key: str, default: str = "") -> str:
    """Retrieve configuration value prioritizing environment variables then .env files,
    falling back safely to Streamlit secrets (st.secrets).
    """
    from src.gemini_client import INVALID_KEY_PLACEHOLDERS

    if key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        keys_to_check = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    else:
        keys_to_check = [key]

    # 1. Check OS environment variables
    for k in keys_to_check:
        env_val = os.getenv(k)
        if env_val and env_val.strip():
            clean = env_val.strip()
            if clean.lower() not in INVALID_KEY_PLACEHOLDERS:
                return clean

    # 2. Try reading directly from candidate .env files
    for p in get_env_paths():
        if p.is_file():
            try:
                file_values = dotenv_values(p)
                for k in keys_to_check:
                    if k in file_values and file_values[k]:
                        val = str(file_values[k]).strip()
                        if val and val.lower() not in INVALID_KEY_PLACEHOLDERS:
                            return val
            except Exception:
                pass

    # 3. Fall back safely to Streamlit secrets (st.secrets) if running in Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and st.secrets is not None:
            for k in keys_to_check:
                if k in st.secrets:
                    val = str(st.secrets[k]).strip()
                    if val and val.lower() not in INVALID_KEY_PLACEHOLDERS:
                        return val
            # Check nested sections e.g. st.secrets["gemini"]["api_key"]
            for section in ["gemini", "general", "api"]:
                if section in st.secrets and isinstance(st.secrets[section], dict):
                    sec = st.secrets[section]
                    for k in keys_to_check:
                        if k.lower() in sec:
                            val = str(sec[k.lower()]).strip()
                            if val and val.lower() not in INVALID_KEY_PLACEHOLDERS:
                                return val
                    if "api_key" in sec:
                        val = str(sec["api_key"]).strip()
                        if val and val.lower() not in INVALID_KEY_PLACEHOLDERS:
                            return val
    except Exception:
        pass

    return default


SUPPORTED_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


@dataclass
class Settings:
    # LLM Settings
    google_api_key: Optional[str] = None
    gemini_model: str = "gemini-3.5-flash-lite"

    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector DB Settings
    chroma_persist_dir: str = str(PROJECT_ROOT / "chroma_db")

    # Chunking & Retrieval Parameters
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 4

    # Storage Paths
    input_data_dir: Path = PROJECT_ROOT / "Input_Data"
    evaluation_results_dir: Path = PROJECT_ROOT / "Evaluation_Results"

    def __post_init__(self):
        # Dynamically populate from environment/secrets if not explicitly provided
        if self.google_api_key is None:
            self.google_api_key = _get_secret_or_env("GOOGLE_API_KEY", "")
        env_model = _get_secret_or_env("GEMINI_MODEL", "")
        if env_model:
            self.gemini_model = env_model
        env_emb = _get_secret_or_env("EMBEDDING_MODEL", "")
        if env_emb and (not self.embedding_model or self.embedding_model == "all-MiniLM-L6-v2"):
            self.embedding_model = env_emb
        env_chroma = _get_secret_or_env("CHROMA_PERSIST_DIR", "")
        if env_chroma and self.chroma_persist_dir == str(PROJECT_ROOT / "chroma_db"):
            self.chroma_persist_dir = env_chroma
        try:
            self.chunk_size = int(_get_secret_or_env("CHUNK_SIZE", str(self.chunk_size)))
        except ValueError:
            pass
        try:
            self.chunk_overlap = int(_get_secret_or_env("CHUNK_OVERLAP", str(self.chunk_overlap)))
        except ValueError:
            pass
        try:
            self.retrieval_top_k = int(_get_secret_or_env("RETRIEVAL_TOP_K", str(self.retrieval_top_k)))
        except ValueError:
            pass

    @property
    def is_gemini_configured(self) -> bool:
        """Check if a valid, non-placeholder API key is present."""
        key = (self.google_api_key or "").strip()
        if not key:
            return False
        invalid_placeholders = {
            "your_api_key_here",
            "your_google_api_key_here",
            "your_gemini_api_key_here",
            "paste_your_new_key_here",
            "replace_with_a_supported_model_name",
            "your_key_here",
            "enter_your_key_here",
            "your_local_api_key",
        }
        if key.lower() in invalid_placeholders:
            return False
        return len(key) >= 15

    @property
    def masked_key_status(self) -> str:
        """Return safe key presence status without ever exposing secret values."""
        if self.is_gemini_configured:
            return "Configured (Present)"
        return "Not configured"

    def get_safe_config_summary(self) -> Dict[str, Any]:
        """Return audit-safe configuration summary."""
        return {
            "gemini_api_configured": self.is_gemini_configured,
            "key_status": self.masked_key_status,
            "gemini_model": self.gemini_model,
            "embedding_model": self.embedding_model,
            "chroma_persist_dir": self.chroma_persist_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_top_k": self.retrieval_top_k,
        }


def get_settings() -> Settings:
    """Return fresh Settings instance reading latest environment / secrets."""
    reload_environment()
    return Settings()


from src.gemini_client import (
    test_gemini_connection,
    get_available_gemini_models,
    STATUS_KEY_MISSING,
    STATUS_NOT_TESTED,
    STATUS_CONNECTED,
    STATUS_RATE_LIMITED,
    STATUS_AUTH_FAILED,
    STATUS_MODEL_UNAVAILABLE,
    STATUS_SERVICE_UNAVAILABLE,
    STATUS_CONNECTION_FAILED,
)
