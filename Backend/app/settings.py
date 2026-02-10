from __future__ import annotations

import os
from dataclasses import dataclass


def _getenv(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


@dataclass(frozen=True)
class Settings:
    mongo_uri: str
    mongo_db: str
    cors_origins: list[str]
    openai_api_key: str | None
    openai_model: str


def load_settings() -> Settings:
    mongo_uri = _getenv("MONGO_URI", "mongodb://localhost:27017") or "mongodb://localhost:27017"
    mongo_db = _getenv("MONGO_DB", "tsundere_chat") or "tsundere_chat"

    cors_raw = _getenv("CORS_ORIGINS", "http://localhost:5173") or "http://localhost:5173"
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]

    openai_api_key = _getenv("OPENAI_API_KEY")
    openai_model = _getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    return Settings(
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        cors_origins=cors_origins,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )

