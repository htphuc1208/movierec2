"""Letterboxd adapter for the MovieLens-shaped training pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal local environments.
    def tqdm(iterable, **kwargs):
        return iterable

from recommender.data.tmdb import (
    TMDBClient,
    load_cache,
    normalize_tmdb_payload,
    sanitize_enriched_payload,
    save_cache,
)


LetterboxdSplit = Literal["cf", "raw"]
RatingPolicy = Literal["implicit", "explicit"]

LETTERBOXD_TMDB_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LetterboxdData:
    interactions: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame
    split: LetterboxdSplit


@dataclass(frozen=True)
class LetterboxdPreparedData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    user_mapping: pd.DataFrame
    movie_mapping: pd.DataFrame
    interactions_debug: pd.DataFrame
    summary: dict[str, Any]


def read_letterboxd(raw_dir: str | Path, split: LetterboxdSplit = "cf") -> LetterboxdData:
    raw_dir = Path(raw_dir)
    interactions_name = "interactions_cf.csv" if split == "cf" else "interactions.csv"
    movies_name = "movies_cf.csv" if split == "cf" else "movies_seed.csv"
    interactions = pd.read_csv(raw_dir / interactions_name, encoding="utf-8-sig")
    movies = pd.read_csv(raw_dir / movies_name, encoding="utf-8-sig")
    users_path = raw_dir / "users.csv"
    users = pd.read_csv(users_path, encoding="utf-8-sig") if users_path.exists() else pd.DataFrame()

    required_interactions = {
        "user_id",
        "movie_id",
        "interaction_type",
        "rating",
        "implicit_score",
        "source",
        "watched_date",
        "created_at",
    }
    required_movies = {"movie_id", "title", "year", "movie_url", "created_at"}
    if not required_interactions.issubset(interactions.columns):
        raise ValueError(f"{interactions_name} must include {sorted(required_interactions)}")
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"{movies_name} must include {sorted(required_movies)}")
    return LetterboxdData(interactions=interactions, movies=movies, users=users, split=split)


def materialize_letterboxd(
    raw_dir: str | Path,
    output_dir: str | Path,
    split: LetterboxdSplit = "cf",
    rating_policy: RatingPolicy = "implicit",
    seed: int = 42,
) -> LetterboxdPreparedData:
    """Convert Letterboxd crawler CSVs into a MovieLens-compatible folder."""
    data = read_letterboxd(raw_dir, split=split)
    interactions = data.interactions.copy()
    interactions["explicit_rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions["implicit_score"] = pd.to_numeric(interactions["implicit_score"], errors="coerce")

    if rating_policy == "explicit":
        usable = interactions.loc[interactions["interaction_type"].eq("rating") & interactions["explicit_rating"].notna()].copy()
        usable["training_rating"] = usable["explicit_rating"]
    else:
        usable = interactions.loc[interactions["implicit_score"].notna()].copy()
        usable["training_rating"] = usable["implicit_score"]

    if usable.empty:
        raise ValueError("No usable Letterboxd interactions after applying rating policy")
    usable = usable.reset_index(drop=True)

    user_ids = usable["user_id"].drop_duplicates().tolist()
    movie_ids = usable["movie_id"].drop_duplicates().tolist()
    user_to_int = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
    movie_to_int = {movie_id: idx + 1 for idx, movie_id in enumerate(movie_ids)}

    usable["userId"] = usable["user_id"].map(user_to_int).astype(int)
    usable["movieId"] = usable["movie_id"].map(movie_to_int).astype(int)
    watched_datetime = pd.to_datetime(usable["watched_date"], errors="coerce", utc=True)
    usable["watched_timestamp"] = np.where(
        watched_datetime.notna(),
        watched_datetime.astype("int64") // 10**9,
        np.nan,
    )
    usable["timestamp_source"] = np.where(watched_datetime.notna(), "watched_date", "missing")
    usable["timestamp"] = _synthetic_random_timestamp(usable[["userId", "movieId"]], seed=seed)

    ratings = usable[["userId", "movieId", "training_rating", "timestamp"]].rename(columns={"training_rating": "rating"})
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    movies = _build_movies_table(data.movies, movie_to_int)
    user_mapping = _build_user_mapping(data.users, user_to_int)
    movie_mapping = _build_movie_mapping(data.movies, movie_to_int)
    interactions_debug = usable[
        [
            "userId",
            "movieId",
            "user_id",
            "movie_id",
            "interaction_type",
            "source",
            "explicit_rating",
            "implicit_score",
            "training_rating",
            "watched_date",
            "watched_timestamp",
            "timestamp",
            "timestamp_source",
            "created_at",
        ]
    ].sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    summary = _build_summary(data, usable, ratings, split=split, rating_policy=rating_policy, seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_dir / "ratings.csv", index=False)
    movies.to_csv(output_dir / "movies.csv", index=False)
    pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"]).to_csv(output_dir / "links.csv", index=False)
    user_mapping.to_csv(output_dir / "letterboxd_user_mapping.csv", index=False)
    movie_mapping.to_csv(output_dir / "letterboxd_movie_mapping.csv", index=False)
    interactions_debug.to_csv(output_dir / "letterboxd_interactions_debug.csv", index=False)
    (output_dir / "letterboxd_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return LetterboxdPreparedData(
        ratings=ratings,
        movies=movies,
        user_mapping=user_mapping,
        movie_mapping=movie_mapping,
        interactions_debug=interactions_debug,
        summary=summary,
    )


def build_base_letterboxd_catalog(output_dir: str | Path) -> pd.DataFrame:
    """Build a catalog parquet with Letterboxd fields and empty TMDb metadata."""
    output_dir = Path(output_dir)
    movies = pd.read_csv(output_dir / "movies.csv")
    mapping = pd.read_csv(output_dir / "letterboxd_movie_mapping.csv")
    catalog = movies.merge(mapping, on=["movieId", "title"], how="left")
    defaults = sanitize_enriched_payload({})
    for key, value in defaults.items():
        if key not in catalog.columns:
            catalog[key] = value
    return catalog


def enrich_letterboxd_catalog(
    output_dir: str | Path,
    client: TMDBClient,
    cache_path: str | Path,
    limit: int | None = None,
    sleep_seconds: float = 0.5,
    min_match_score: int = 75,
) -> pd.DataFrame:
    """Search TMDb by Letterboxd title/year and emit the shared enriched catalog schema."""
    output_dir = Path(output_dir)
    mapping = pd.read_csv(output_dir / "letterboxd_movie_mapping.csv")
    movies = pd.read_csv(output_dir / "movies.csv")
    catalog_input = movies.merge(mapping, on=["movieId", "title"], how="left")
    if limit:
        catalog_input = catalog_input.head(limit)

    cache_path = Path(cache_path)
    cache = load_cache(cache_path)
    rows: list[dict[str, Any]] = []

    for row in tqdm(catalog_input.itertuples(index=False), total=len(catalog_input), desc="Letterboxd TMDb enrichment"):
        base = {
            "movieId": int(row.movieId),
            "title": row.title,
            "genres": row.genres,
            "letterboxd_movie_id": row.letterboxd_movie_id,
            "letterboxd_title": row.letterboxd_title,
            "letterboxd_year": "" if pd.isna(row.letterboxd_year) else str(row.letterboxd_year),
            "letterboxd_movie_url": row.letterboxd_movie_url if pd.notna(row.letterboxd_movie_url) else "",
            "title_for_tmdb": row.title_for_tmdb,
        }
        cache_key = f"letterboxd:{base['letterboxd_movie_id']}"
        needs_refresh = (
            cache_key not in cache
            or cache[cache_key].get("schema_version") != LETTERBOXD_TMDB_CACHE_SCHEMA_VERSION
        )
        if needs_refresh:
            cache[cache_key] = _fetch_letterboxd_tmdb_match(client, base, min_match_score)
            save_cache(cache_path, cache)
            if sleep_seconds:
                time.sleep(sleep_seconds)

        cached = cache[cache_key]
        enriched = sanitize_enriched_payload(cached.get("data", normalize_tmdb_payload({})))
        rows.append(
            {
                **base,
                **enriched,
                "enrich_status": cached.get("status", "error"),
                "enrich_match_score": cached.get("match_score", 0),
                "tmdb_match_title": cached.get("tmdb_match_title", ""),
            }
        )

    catalog = pd.DataFrame(rows)
    output_path = output_dir / "movie_catalog_enriched.parquet"
    catalog.to_parquet(output_path, index=False)
    return catalog


def _synthetic_random_timestamp(ids: pd.DataFrame, seed: int) -> np.ndarray:
    timestamps = np.empty(len(ids), dtype=np.int64)
    for user_id, group in ids.groupby("userId", sort=True):
        positions = group.index.to_numpy()
        rng = np.random.default_rng(_stable_user_seed(seed, int(user_id)))
        shuffled_ranks = rng.permutation(len(positions))
        timestamps[positions] = shuffled_ranks + 1
    return timestamps


def _stable_user_seed(seed: int, user_id: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{user_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _build_movies_table(movies_raw: pd.DataFrame, movie_to_int: dict[str, int]) -> pd.DataFrame:
    mapping = _build_movie_mapping(movies_raw, movie_to_int)
    movies = mapping[["movieId", "title"]].copy()
    movies["genres"] = "(no genres listed)"
    return movies.sort_values("movieId").reset_index(drop=True)


def _build_user_mapping(users_raw: pd.DataFrame, user_to_int: dict[str, int]) -> pd.DataFrame:
    mapping = pd.DataFrame({"letterboxd_user_id": list(user_to_int.keys()), "userId": list(user_to_int.values())})
    if not users_raw.empty:
        users = users_raw.rename(columns={"user_id": "letterboxd_user_id"}).copy()
        keep_cols = [
            "letterboxd_user_id",
            "username",
            "display_name",
            "profile_url",
            "films_count",
            "following_count",
            "followers_count",
            "created_at",
        ]
        users = users[[col for col in keep_cols if col in users.columns]]
        mapping = mapping.merge(users, on="letterboxd_user_id", how="left")
    return mapping.sort_values("userId").reset_index(drop=True)


def _build_movie_mapping(movies_raw: pd.DataFrame, movie_to_int: dict[str, int]) -> pd.DataFrame:
    mapping = pd.DataFrame({"letterboxd_movie_id": list(movie_to_int.keys()), "movieId": list(movie_to_int.values())})
    movies = movies_raw.rename(columns={"movie_id": "letterboxd_movie_id"}).copy()
    movies["letterboxd_title"] = movies["title"].fillna("")
    movies["letterboxd_year"] = movies.apply(lambda row: _extract_year(row.get("title"), row.get("year")), axis=1)
    movies["title_for_tmdb"] = movies["letterboxd_title"].map(_strip_trailing_year)
    movies["title"] = movies["letterboxd_title"]
    movies = movies.rename(columns={"movie_url": "letterboxd_movie_url", "created_at": "letterboxd_created_at"})
    keep_cols = [
        "letterboxd_movie_id",
        "title",
        "letterboxd_title",
        "letterboxd_year",
        "title_for_tmdb",
        "letterboxd_movie_url",
        "letterboxd_created_at",
    ]
    return mapping.merge(movies[keep_cols], on="letterboxd_movie_id", how="left").sort_values("movieId").reset_index(drop=True)


def _extract_year(title: Any, year: Any) -> str:
    if pd.notna(year):
        text = str(year).strip()
        if text.endswith(".0"):
            text = text[:-2]
        if text.isdigit():
            return text
    match = re.search(r"\((\d{4})\)\s*$", str(title or ""))
    return match.group(1) if match else ""


def _strip_trailing_year(title: Any) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title or "")).strip()


def _normalize_title(title: str) -> str:
    text = _strip_trailing_year(title).lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_similarity(left: str, right: str) -> int:
    left_norm = _normalize_title(left)
    right_norm = _normalize_title(right)
    if not left_norm or not right_norm:
        return 0
    if left_norm == right_norm:
        return 100
    return int(SequenceMatcher(None, left_norm, right_norm).ratio() * 100)


def _year_score(letterboxd_year: str, release_date: str) -> int:
    if not letterboxd_year or not release_date:
        return 0
    try:
        diff = abs(int(letterboxd_year) - int(str(release_date)[:4]))
    except ValueError:
        return 0
    if diff == 0:
        return 20
    if diff == 1:
        return 10
    return -20


def _fetch_letterboxd_tmdb_match(client: TMDBClient, base: dict[str, Any], min_match_score: int) -> dict[str, Any]:
    query = str(base.get("title_for_tmdb") or base.get("title") or "").strip()
    year = str(base.get("letterboxd_year") or "").strip()
    if not query:
        return _cache_payload("error", 0, "", normalize_tmdb_payload({}), "missing title")

    try:
        search = client.search_movie(query, year=year or None)
        results = search.get("results", [])
        if not results and year:
            search = client.search_movie(query)
            results = search.get("results", [])
        best, score = _pick_best_tmdb_result(results, query, year)
        if not best or score < min_match_score:
            return _cache_payload("not_found", score, "", normalize_tmdb_payload({}), "")
        details = client.movie_details(int(best["id"]))
        return _cache_payload("matched", score, best.get("title", ""), normalize_tmdb_payload(details), "")
    except requests.RequestException as exc:
        return _cache_payload("error", 0, "", normalize_tmdb_payload({}), str(exc))


def _pick_best_tmdb_result(results: list[dict[str, Any]], title: str, year: str) -> tuple[dict[str, Any] | None, int]:
    best: dict[str, Any] | None = None
    best_score = 0
    for candidate in results[:5]:
        title_score = max(
            _title_similarity(title, candidate.get("title", "")),
            _title_similarity(title, candidate.get("original_title", "")),
        )
        score = title_score + _year_score(year, candidate.get("release_date", ""))
        if score > best_score:
            best = candidate
            best_score = score
    return best, best_score


def _cache_payload(status: str, match_score: int, tmdb_match_title: str, data: dict[str, Any], error: str) -> dict[str, Any]:
    payload = {
        "ok": status == "matched",
        "status": status,
        "schema_version": LETTERBOXD_TMDB_CACHE_SCHEMA_VERSION,
        "match_score": int(match_score),
        "tmdb_match_title": tmdb_match_title,
        "data": data,
    }
    if error:
        payload["error"] = error
    return payload


def _build_summary(
    data: LetterboxdData,
    usable: pd.DataFrame,
    ratings: pd.DataFrame,
    split: LetterboxdSplit,
    rating_policy: RatingPolicy,
    seed: int,
) -> dict[str, Any]:
    watched_dates = pd.to_datetime(data.interactions["watched_date"], errors="coerce", utc=True)
    return {
        "source": "letterboxd",
        "split": split,
        "rating_policy": rating_policy,
        "random_seed": seed,
        "split_strategy": "synthetic_random_per_user",
        "note": "created_at is crawler time and is not used as behavior time.",
        "raw_interactions": int(len(data.interactions)),
        "usable_interactions": int(len(usable)),
        "output_ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "movies": int(ratings["movieId"].nunique()),
        "watched_date_parseable": int(watched_dates.notna().sum()),
        "watched_date_parseable_pct": float(watched_dates.notna().mean() * 100.0),
        "interaction_type_counts": data.interactions["interaction_type"].value_counts().to_dict(),
    }
