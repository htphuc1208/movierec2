"""TMDb enrichment client with resumable JSON cache."""

from __future__ import annotations

import json
import random
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
TMDB_CACHE_SCHEMA_VERSION = 3

ENRICHED_TEXT_FIELDS = [
    "overview",
    "tmdb_genres",
    "keywords",
    "director",
    "writers",
    "cast",
    "collection",
    "production_companies",
    "production_countries",
    "original_language",
    "release_year",
]

ENRICHED_FIELDS = [
    "tmdb_id",
    "overview",
    "poster_url",
    "release_date",
    "release_year",
    "tmdb_genres",
    "keywords",
    "popularity",
    "vote_average",
    "vote_count",
    "runtime_minutes",
    "original_language",
    "production_countries",
    "production_companies",
    "collection",
    "director",
    "writers",
    "cast",
]


@dataclass
class TMDBClient:
    api_key: str
    language: str = "en-US"
    timeout: float = 15.0
    max_retries: int = 5
    retry_backoff: float = 1.5
    base_url: str = "https://api.themoviedb.org/3"
    user_agent: str = "movierec3/0.1 (+https://developer.themoviedb.org/docs)"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TMDB_API_KEY is required for TMDb enrichment")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            }
        )

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        params = {"api_key": self.api_key, "language": self.language, **params}
        url = f"{self.base_url}{path}"
        last_error: requests.RequestException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    if attempt < self.max_retries:
                        self._sleep_before_retry(attempt, retry_after)
                        continue
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise
                self._sleep_before_retry(attempt)
            except requests.RequestException:
                raise
        if last_error:
            raise last_error
        raise RuntimeError(f"TMDb request failed unexpectedly: {url}")

    def _sleep_before_retry(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except ValueError:
                pass
        delay = min(60.0, self.retry_backoff * (2**attempt))
        time.sleep(delay + random.uniform(0.0, 0.25))

    def movie(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}")

    def movie_details(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}", append_to_response="credits,keywords")

    def search_movie(self, query: str, year: str | int | None = None, include_adult: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "include_adult": str(include_adult).lower()}
        if year:
            params["year"] = str(year)
        return self._get("/search/movie", **params)

    def credits(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}/credits")

    def keywords(self, tmdb_id: int) -> dict[str, Any]:
        return self._get(f"/movie/{int(tmdb_id)}/keywords")

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


def _extract_writers(credits: dict[str, Any], limit: int = 5) -> list[str]:
    writer_jobs = {"Writer", "Screenplay", "Story", "Novel", "Characters"}
    names: list[str] = []
    for crew in credits.get("crew", []):
        if crew.get("job") in writer_jobs and crew.get("name") and crew.get("name") not in names:
            names.append(crew["name"])
        if len(names) >= limit:
            break
    return names

# hàm này dùng để nối tên các thể loại phim, từ khóa, diễn viên, đạo diễn,... thành một chuỗi duy nhất phân cách bằng dấu "|", nếu có limit thì chỉ lấy số lượng tên giới hạn đó, nếu key không tồn tại trong value thì trả về chuỗi rỗng
def _pipe_names(values: list[dict[str, Any]], key: str = "name", limit: int | None = None) -> str:
    names = [str(value.get(key, "")).strip() for value in values if value.get(key)]
    if limit is not None:
        names = names[:limit]
    return "|".join(names)


def _release_year(release_date: str | None) -> str:
    if not release_date:
        return ""
    return str(release_date)[:4] if str(release_date)[:4].isdigit() else ""


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def normalize_tmdb_payload(
    movie: dict[str, Any],
    credits: dict[str, Any] | None = None,
    keywords: dict[str, Any] | None = None,
) -> dict[str, Any]:
    credits = credits or {}
    if not credits and isinstance(movie.get("credits"), dict):
        credits = movie["credits"]
    keywords = keywords or {}
    if not keywords and isinstance(movie.get("keywords"), dict):
        keywords = movie["keywords"]
    release_date = movie.get("release_date") or ""
    collection = movie.get("belongs_to_collection") or {}
    return {
        "tmdb_id": movie.get("id"),
        "overview": movie.get("overview") or "",
        "poster_url": TMDBClient.poster_url(movie.get("poster_path")),
        "release_date": release_date,
        "release_year": _release_year(release_date),
        "tmdb_genres": _pipe_names(movie.get("genres", [])),
        "keywords": _pipe_names(keywords.get("keywords", []), limit=20),
        "popularity": float(movie.get("popularity") or 0.0),
        "vote_average": float(movie.get("vote_average") or 0.0),
        "vote_count": _int_or_zero(movie.get("vote_count")),
        "runtime_minutes": _int_or_zero(movie.get("runtime")),
        "original_language": movie.get("original_language") or "",
        "production_countries": _pipe_names(movie.get("production_countries", [])),
        "production_companies": _pipe_names(movie.get("production_companies", []), limit=5),
        "collection": collection.get("name") or "",
        "director": _extract_director(credits) or "",
        "writers": "|".join(_extract_writers(credits)),
        "cast": "|".join(_extract_cast(credits)),
    }

# đảm bảo rằng payload đã được chuẩn hóa chỉ chứa các trường trong ENRICHED_FIELDS, nếu trường nào không có trong payload 
# thì sẽ lấy giá trị mặc định từ normalize_tmdb_payload({}), đặc biệt là release_year sẽ được lấy từ release_date nếu release_year không có hoặc rỗng
def sanitize_enriched_payload(payload: dict[str, Any]) -> dict[str, Any]:
    defaults = normalize_tmdb_payload({})
    sanitized = {field: payload.get(field, defaults[field]) for field in ENRICHED_FIELDS}
    sanitized["release_year"] = sanitized.get("release_year") or _release_year(sanitized.get("release_date"))
    return sanitized


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
        needs_refresh = cache_key not in cache or cache[cache_key].get("schema_version") != TMDB_CACHE_SCHEMA_VERSION
        if needs_refresh:
            try:
                movie_payload = client.movie_details(base["tmdb_id"])
                cache[cache_key] = {
                    "ok": True,
                    "schema_version": TMDB_CACHE_SCHEMA_VERSION,
                    "data": normalize_tmdb_payload(movie_payload),
                }
            except requests.RequestException as exc:
                cache[cache_key] = {
                    "ok": False,
                    "schema_version": TMDB_CACHE_SCHEMA_VERSION,
                    "error": str(exc),
                    "data": normalize_tmdb_payload({}),
                }
            save_cache(cache_path, cache)
            if sleep_seconds:
                time.sleep(sleep_seconds)

        enriched = sanitize_enriched_payload(cache[cache_key].get("data", normalize_tmdb_payload({})))
        rows.append({**base, **enriched})

    return pd.DataFrame(rows)
# vai trò: dùng cho sbert , dùng làm đặc trưng số, dùng cho UI
