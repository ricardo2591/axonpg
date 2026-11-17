"""Configuration loading, validation, and multi-provider presets for AxonPG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

StorageMode = Literal["docker", "lite"]
EmbedProviderName = Literal[
    "fastembed",
    "ollama",
    "nvidia",
    "openrouter",
    "gemini",
    "openai",
    "custom",
    "none",
]
LLMProviderName = Literal[
    "none",
    "openrouter",
    "nvidia",
    "ollama",
    "gemini",
    "openai",
    "custom",
]


@dataclass(frozen=True)
class ProviderPreset:
    """Default connection parameters for a supported embedding or LLM provider."""

    name: str
    default_model: str
    default_dim: int
    base_url: str
    requires_api_key: bool


EMBEDDING_PRESETS: dict[str, ProviderPreset] = {
    "fastembed": ProviderPreset(
        name="fastembed",
        default_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        default_dim=384,
        base_url="",
        requires_api_key=False,
    ),
    "ollama": ProviderPreset(
        name="ollama",
        default_model="bge-m3",
        default_dim=1024,
        base_url="http://127.0.0.1:11434",
        requires_api_key=False,
    ),
    "nvidia": ProviderPreset(
        name="nvidia",
        default_model="baai/bge-m3",
        default_dim=1024,
        base_url="https://integrate.api.nvidia.com/v1",
        requires_api_key=True,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        default_model="openai/text-embedding-3-small",
        default_dim=1536,
        base_url="https://openrouter.ai/api/v1",
        requires_api_key=True,
    ),
    "gemini": ProviderPreset(
        name="gemini",
        default_model="gemini-embedding-001",
        default_dim=768,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        requires_api_key=True,
    ),
    "openai": ProviderPreset(
        name="openai",
        default_model="text-embedding-3-small",
        default_dim=1536,
        base_url="https://api.openai.com/v1",
        requires_api_key=True,
    ),
    "custom": ProviderPreset(
        name="custom",
        default_model="text-embedding-3-small",
        default_dim=768,
        base_url="http://127.0.0.1:8000/v1",
        requires_api_key=False,
    ),
    "none": ProviderPreset(
        name="none",
        default_model="none",
        default_dim=384,
        base_url="",
        requires_api_key=False,
    ),
}

LLM_PRESETS: dict[str, ProviderPreset] = {
    "none": ProviderPreset(
        name="none",
        default_model="deterministic",
        default_dim=0,
        base_url="",
        requires_api_key=False,
    ),
    "openrouter": ProviderPreset(
        name="openrouter",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        default_dim=0,
        base_url="https://openrouter.ai/api/v1",
        requires_api_key=True,
    ),
    "nvidia": ProviderPreset(
        name="nvidia",
        default_model="meta/llama-3.1-70b-instruct",
        default_dim=0,
        base_url="https://integrate.api.nvidia.com/v1",
        requires_api_key=True,
    ),
    "ollama": ProviderPreset(
        name="ollama",
        default_model="qwen2.5:7b",
        default_dim=0,
        base_url="http://127.0.0.1:11434",
        requires_api_key=False,
    ),
    "gemini": ProviderPreset(
        name="gemini",
        default_model="gemini-2.5-flash",
        default_dim=0,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        requires_api_key=True,
    ),
    "openai": ProviderPreset(
        name="openai",
        default_model="gpt-4o-mini",
        default_dim=0,
        base_url="https://api.openai.com/v1",
        requires_api_key=True,
    ),
    "custom": ProviderPreset(
        name="custom",
        default_model="local-model",
        default_dim=0,
        base_url="http://127.0.0.1:8000/v1",
        requires_api_key=False,
    ),
}


class AxonConfig(BaseModel):
    """Runtime settings for AxonPG loaded from environment variables or explicit kwargs."""

    mode: StorageMode = "docker"
    namespace: str = Field(default="default", min_length=1, max_length=64)
    lang: str = "auto"

    sqlite_path: str = "~/.axonpg/axonpg.db"

    db_host: str = "127.0.0.1"
    db_port: int = Field(default=5435, ge=1, le=65535)
    db_name: str = "axonpg"
    db_user: str = "axonpg"
    db_password: str = "axonpg_change_me"

    embed_provider: EmbedProviderName = "fastembed"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = Field(default=384, ge=32, le=4096)
    embed_api_key: str = ""
    embed_base_url: str = ""

    llm_provider: LLMProviderName = "none"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_api_key: str = ""
    llm_base_url: str = ""

    @classmethod
    def from_env(cls, overrides: dict[str, Any] | None = None) -> AxonConfig:
        """Load configuration from environment variables with preset defaults applied."""
        raw_embed_prov = os.getenv("AXONPG_EMBED_PROVIDER", "fastembed").strip().lower()
        embed_preset = EMBEDDING_PRESETS.get(raw_embed_prov, EMBEDDING_PRESETS["fastembed"])

        raw_llm_prov = os.getenv("AXONPG_LLM_PROVIDER", "none").strip().lower()
        llm_preset = LLM_PRESETS.get(raw_llm_prov, LLM_PRESETS["none"])

        data: dict[str, Any] = {
            "mode": os.getenv("AXONPG_MODE", "docker").strip().lower(),
            "namespace": os.getenv("AXONPG_NAMESPACE", "default").strip() or "default",
            "lang": os.getenv("AXONPG_LANG", "auto").strip().lower(),
            "sqlite_path": os.getenv("AXONPG_SQLITE_PATH", "~/.axonpg/axonpg.db"),
            "db_host": os.getenv("AXONPG_DB_HOST", "127.0.0.1"),
            "db_port": int(os.getenv("AXONPG_DB_PORT", "5435")),
            "db_name": os.getenv("AXONPG_DB_NAME", "axonpg"),
            "db_user": os.getenv("AXONPG_DB_USER", "axonpg"),
            "db_password": os.getenv("AXONPG_DB_PASSWORD", "axonpg_change_me"),
            "embed_provider": raw_embed_prov,
            "embed_model": os.getenv("AXONPG_EMBED_MODEL") or embed_preset.default_model,
            "embed_dim": int(os.getenv("AXONPG_EMBED_DIM") or embed_preset.default_dim),
            "embed_api_key": os.getenv("AXONPG_EMBED_API_KEY", ""),
            "embed_base_url": os.getenv("AXONPG_EMBED_BASE_URL") or embed_preset.base_url,
            "llm_provider": raw_llm_prov,
            "llm_model": os.getenv("AXONPG_LLM_MODEL") or llm_preset.default_model,
            "llm_api_key": os.getenv("AXONPG_LLM_API_KEY", ""),
            "llm_base_url": os.getenv("AXONPG_LLM_BASE_URL") or llm_preset.base_url,
        }
        if overrides:
            data.update(overrides)
        return cls(**data)

    def resolved_sqlite_path(self) -> Path:
        """Expand user home directory for the SQLite file path."""
        return Path(self.sqlite_path).expanduser().resolve()

    def postgres_dsn(self) -> str:
        """Build a PostgreSQL connection URI (not for unmasked logging)."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def redacted_dict(self) -> dict[str, Any]:
        """Return settings dictionary with sensitive keys masked for safe display."""
        payload = self.model_dump()
        for secret_field in ("db_password", "embed_api_key", "llm_api_key"):
            val = payload.get(secret_field, "")
            payload[secret_field] = "***" if val else ""
        return payload

    def to_env_text(self) -> str:
        """Render settings as a clean `.env` file."""
        lines = [
            "# Generated by AxonPG CLI",
            f"AXONPG_MODE={self.mode}",
            f"AXONPG_NAMESPACE={self.namespace}",
            f"AXONPG_LANG={self.lang}",
            f"AXONPG_SQLITE_PATH={self.sqlite_path}",
            f"AXONPG_DB_HOST={self.db_host}",
            f"AXONPG_DB_PORT={self.db_port}",
            f"AXONPG_DB_NAME={self.db_name}",
            f"AXONPG_DB_USER={self.db_user}",
            f"AXONPG_DB_PASSWORD={self.db_password}",
            f"AXONPG_EMBED_PROVIDER={self.embed_provider}",
            f"AXONPG_EMBED_MODEL={self.embed_model}",
            f"AXONPG_EMBED_DIM={self.embed_dim}",
            f"AXONPG_EMBED_API_KEY={self.embed_api_key}",
            f"AXONPG_EMBED_BASE_URL={self.embed_base_url}",
            f"AXONPG_LLM_PROVIDER={self.llm_provider}",
            f"AXONPG_LLM_MODEL={self.llm_model}",
            f"AXONPG_LLM_API_KEY={self.llm_api_key}",
            f"AXONPG_LLM_BASE_URL={self.llm_base_url}",
            "",
        ]
        return "\n".join(lines)
