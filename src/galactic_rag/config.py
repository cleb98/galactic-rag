from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings

load_dotenv()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ProviderConfig(BaseModel):
    api_key_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...]
    embedding_envs: tuple[str, ...]
    llm_envs: tuple[str, ...]
    model_config = {"frozen": True}


DEFAULT_EMBED_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-3-large-512": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-004": 1536,
    "text-embedding-4-small": 1536,
    "text-embedding-4-large": 3072,
    "gemini-embedding-001": 3072,
    "models/embedding-001": 3072,
}


PROVIDER_DEFAULTS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        api_key_envs=("OPENAI_API_KEY",),
        base_url_envs=("OPENAI_BASE_URL",),
        embedding_envs=("OPENAI_EMBEDDING_MODEL",),
        llm_envs=("OPENAI_MODEL",),
    ),
    "gemini": ProviderConfig(
        api_key_envs=("GEMINI_API_KEY",),
        base_url_envs=("GEMINI_BASE_URL",),
        embedding_envs=("GEMINI_EMBEDDING_MODEL",),
        llm_envs=("GEMINI_MODEL",),
    ),
}


class Settings(BaseSettings):
    """Central configuration loaded from environment variables or defaults."""

    project_root: Path = Field(default_factory=_default_project_root)
    dataset_root: Path = Field(
        default_factory=lambda: _default_project_root() / "Dataset"
    )
    solution_data_dir: Path = Field(
        default_factory=lambda: _default_project_root() / "data"
    )

    # Vectorstore / ingestion settings
    vectorstore_path: Path = Field(
        default_factory=lambda: _default_project_root() / "data" / "vectorstore"
    )
    qdrant_host: str | None = Field(default=None, env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: str | None = Field(default=None, env="QDRANT_API_KEY")
    qdrant_https: bool = Field(default=False, env="QDRANT_HTTPS")
    qdrant_collection: str = Field(default="galactic_rag_collection", env="QDRANT_COLLECTION")
    chunk_max_chars: int = 1200
    embedding_dimensions_map: dict[str, int] = Field(
        default_factory=lambda: DEFAULT_EMBED_DIMENSIONS.copy()
    )
    docling_json_output_dir: Path = Field(
        default_factory=lambda: _default_project_root() / "data" / "docling_json"
    )

    # Model defaults
    provider: Literal["openai", "gemini"] = Field(env="PROVIDER")

    def _from_envs(self, *, env_candidates: tuple[str, ...]) -> str:
        for env_name in env_candidates:
            value = os.getenv(env_name)
            if value:
                return value
        raise ValueError(
            f"Missing required environment variables {env_candidates} for provider '{self.provider}'."
        )

    @computed_field  # type: ignore[misc]
    @property
    def api_key(self) -> str:
        provider_defaults = PROVIDER_DEFAULTS[self.provider]
        return self._from_envs(env_candidates=provider_defaults.api_key_envs)

    @computed_field  # type: ignore[misc]
    @property
    def base_url(self) -> str:
        provider_defaults = PROVIDER_DEFAULTS[self.provider]
        return self._from_envs(
            env_candidates=provider_defaults.base_url_envs,
        )

    @computed_field  # type: ignore[misc]
    @property
    def embedding_model(self) -> str:
        provider_defaults = PROVIDER_DEFAULTS[self.provider]
        return self._from_envs(
            env_candidates=provider_defaults.embedding_envs,
        )

    @computed_field  # type: ignore[misc]
    @property
    def llm_model(self) -> str:
        provider_defaults = PROVIDER_DEFAULTS[self.provider]
        return self._from_envs(
            env_candidates=provider_defaults.llm_envs,
        )

    def embedding_dimensions(self) -> int:
        dims = self.embedding_dimensions_map.get(self.embedding_model)
        if dims:
            return dims
        for prefix, value in self.embedding_dimensions_map.items():
            if self.embedding_model.startswith(prefix):
                return value
        raise ValueError(  # nosec B608: configuration error message, not SQL construction
            f"Unknown embedding dimensions for model '{self.embedding_model}'. "
            "Update embedding_dimensions_map or set EMBEDDING_DIMENSIONS via env/config."
        )

    class Config:
        env_prefix = ""
        extra = "ignore"

    @computed_field  # type: ignore[misc]
    @property
    def knowledge_base_dir(self) -> Path:
        return self.dataset_root / "knowledge_base"

    @computed_field  # type: ignore[misc]
    @property
    def domande_csv(self) -> Path:
        return self.dataset_root / "domande.csv"

    @computed_field  # type: ignore[misc]
    @property
    def dish_mapping_json(self) -> Path:
        return self.dataset_root / "ground_truth" / "dish_mapping.json"

    def ensure_directories(self) -> None:
        self.solution_data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
