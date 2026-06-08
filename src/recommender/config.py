"""Project configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_if_available(path: Path | None = None) -> None:
    """Load .env when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path or PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    ratings_store_path: Path = PROJECT_ROOT / "artifacts" / "runtime" / "user_ratings.csv"
    tmdb_api_key: str | None = None
    tmdb_language: str = "vi-VN"
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    api_url: str = "http://localhost:8000"
    openai_api_key: str | None = None
    chat_model: str = "gpt-4.1-mini"


def get_settings() -> AppSettings:
    load_dotenv_if_available()
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
    return AppSettings(
        raw_dir=Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw")),
        processed_dir=Path(os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")),
        artifacts_dir=artifacts_dir,
        ratings_store_path=Path(os.getenv("RATINGS_STORE_PATH", artifacts_dir / "runtime" / "user_ratings.csv")),
        tmdb_api_key=os.getenv("TMDB_API_KEY"),
        tmdb_language=os.getenv("TMDB_LANGUAGE", "vi-VN"),
        tmdb_base_url=os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3"),
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
    )
