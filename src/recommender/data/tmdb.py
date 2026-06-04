"""TMDb enrichment client with resumable JSON cache."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal local environments.
    def tqdm(iterable, **kwargs):
        return iterable


TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"


@dataclass
class TMDBClient:
    api_key: str
    language: str = "vi-VN"
    timeout: float = 15.0
    base_url: str = "https://api.themoviedb.org/3"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TMDB_API_KEY is required for TMDb enrichment")
        self.session = requests.Session()

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        params = {"api_key": self.api_key, "language": self.language, **params}
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def movie(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}")

    def credits(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}/credits")

    @staticmethod
    def poster_url(path: str | None, size: str = "w500") -> str | None:
        if not path:
            return None
        return f"{TMDB_IMAGE_BASE}/{size}{path}"


def load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(cache_path)


def _extract_director(credits: dict[str, Any]) -> str | None:
    for crew in credits.get("crew", []):
        if crew.get("job") == "Director":
            return crew.get("name")
    return None


def _extract_cast(credits: dict[str, Any], limit: int = 5) -> list[str]:
    cast = sorted(credits.get("cast", []), key=lambda row: row.get("order", 9999))
    return [row.get("name", "") for row in cast[:limit] if row.get("name")]


def normalize_tmdb_payload(movie: dict[str, Any], credits: dict[str, Any] | None = None) -> dict[str, Any]:
    credits = credits or {}
    return {
        "tmdb_id": movie.get("id"),
        "overview": movie.get("overview") or "",
        "tagline": movie.get("tagline") or "",
        "poster_url": TMDBClient.poster_url(movie.get("poster_path")),
        "release_date": movie.get("release_date") or "",
        "tmdb_genres": "|".join(genre.get("name", "") for genre in movie.get("genres", []) if genre.get("name")),
        "popularity": float(movie.get("popularity") or 0.0),
        "vote_average": float(movie.get("vote_average") or 0.0),
        "director": _extract_director(credits) or "",
        "cast": "|".join(_extract_cast(credits)),
    }


def enrich_catalog(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    client: TMDBClient,
    cache_path: str | Path,
    limit: int | None = None,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """Enrich MovieLens catalog with TMDb metadata and poster URLs."""
    cache_path = Path(cache_path)
    cache = load_cache(cache_path)
    if links.empty or "tmdbId" not in links.columns:
        raise ValueError("links.csv with tmdbId is required for TMDb enrichment")

    merged = movies.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
    if limit:
        merged = merged.head(limit)

    rows: list[dict[str, Any]] = []
    for row in tqdm(merged.itertuples(index=False), total=len(merged), desc="TMDb enrichment"):
        base = {
            "movieId": int(row.movieId),
            "title": row.title,
            "genres": row.genres,
            "imdb_id": None if pd.isna(row.imdbId) else int(row.imdbId),
            "tmdb_id": None if pd.isna(row.tmdbId) else int(row.tmdbId),
        }
        if base["tmdb_id"] is None:
            rows.append({**base, **normalize_tmdb_payload({})})
            continue

        cache_key = str(base["tmdb_id"])
        if cache_key not in cache:
            try:
                movie_payload = client.movie(base["tmdb_id"])
                credits_payload = client.credits(base["tmdb_id"])
                cache[cache_key] = {
                    "ok": True,
                    "data": normalize_tmdb_payload(movie_payload, credits_payload),
                }
            except requests.HTTPError as exc:
                cache[cache_key] = {"ok": False, "error": str(exc), "data": normalize_tmdb_payload({})}
            save_cache(cache_path, cache)
            if sleep_seconds:
                time.sleep(sleep_seconds)

        enriched = cache[cache_key].get("data", normalize_tmdb_payload({}))
        rows.append({**base, **enriched})

    return pd.DataFrame(rows)
