from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd


PLACEHOLDER_METADATA_COLUMNS = [
    "movieId",
    "overview",
    "tagline",
    "director",
    "cast",
    "poster_url",
    "budget",
    "revenue",
    "genres",
    "release_date",
    "runtime",
    "original_language",
    "production_companies",
    "production_countries",
    "keywords",
    "vote_average",
    "vote_count",
    "popularity",
    "collection_id",
    "collection_name",
    "certification",
    "imdb_id",
    "enrichment_status",
]


def source_files(source: str) -> tuple[str, str]:
    if source == "cf":
        return "movies_cf.csv", "ratings_cf.csv"
    if source == "full":
        return "movies_seed.csv", "ratings.csv"
    raise ValueError(f"Unsupported Letterboxd source: {source}")


def prepare_letterboxd(raw_dir: str | Path, output_dir: str | Path, source: str = "full") -> dict[str, Any]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movies_file, ratings_file = source_files(source)
    movies_raw = pd.read_csv(raw_path / movies_file)
    ratings_raw = pd.read_csv(raw_path / ratings_file)

    require_columns(movies_raw, ["movie_id", "title"], movies_file)
    require_columns(ratings_raw, ["user_id", "movie_id", "rating"], ratings_file)
    for column in ["movie_url", "year"]:
        if column not in movies_raw.columns:
            movies_raw[column] = ""

    movies_raw = movies_raw.dropna(subset=["movie_id", "title"]).drop_duplicates("movie_id", keep="last").copy()
    ratings_raw = ratings_raw.dropna(subset=["user_id", "movie_id"]).copy()
    ratings_raw = ratings_raw.loc[ratings_raw["movie_id"].isin(set(movies_raw["movie_id"]))].copy()

    movies_raw["parsed_year"] = movies_raw.apply(lambda row: parsed_year(row.get("year"), row.get("title")), axis=1)
    movies_raw["clean_title"] = movies_raw["title"].map(clean_title)
    movies_raw["canonical_key"] = movies_raw.apply(canonical_key, axis=1)

    canonical = choose_canonical_movies(movies_raw)
    canonical["movieId"] = range(1, len(canonical) + 1)
    key_to_movie_id = canonical.set_index("canonical_key")["movieId"].to_dict()
    raw_to_movie_id = movies_raw.set_index("movie_id")["canonical_key"].map(key_to_movie_id).to_dict()
    user_map = ordered_id_map(ratings_raw["user_id"])

    ratings = pd.DataFrame(
        {
            "userId": ratings_raw["user_id"].map(user_map).astype(int),
            "movieId": ratings_raw["movie_id"].map(raw_to_movie_id).astype(int),
            "rating": normalise_ratings(ratings_raw),
            "timestamp": stable_timestamps(ratings_raw),
        }
    )
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last")
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    movies = pd.DataFrame(
        {
            "movieId": canonical["movieId"].astype(int),
            "title": [
                title_with_year(row.clean_title, row.parsed_year)
                for row in canonical.itertuples()
            ],
            "genres": "(no genres listed)",
            "year": canonical["parsed_year"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True),
        }
    )
    links = pd.DataFrame({"movieId": movies["movieId"], "imdbId": "", "tmdbId": ""})
    tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
    movie_mapping = movies_raw[["movie_id", "canonical_key", "title", "movie_url"]].copy()
    movie_mapping["movieId"] = movie_mapping["movie_id"].map(raw_to_movie_id).astype(int)
    movie_mapping = movie_mapping.rename(columns={"movie_id": "raw_movie_id", "movie_url": "raw_movie_url"})
    user_mapping = pd.DataFrame([{"raw_user_id": raw_id, "userId": user_id} for raw_id, user_id in user_map.items()])

    movies.to_csv(output_path / "movies.csv", index=False)
    ratings.to_csv(output_path / "ratings.csv", index=False)
    tags.to_csv(output_path / "tags.csv", index=False)
    links.to_csv(output_path / "links.csv", index=False)
    movie_mapping.to_csv(output_path / "movie_id_mapping.csv", index=False)
    user_mapping.to_csv(output_path / "user_id_mapping.csv", index=False)
    write_placeholder_enriched(output_path / "enriched_movies.csv", movies)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "raw_dir": str(raw_path),
        "output_dir": str(output_path),
        "raw_movies": int(movies_raw["movie_id"].nunique()),
        "movies": int(len(movies)),
        "raw_ratings": int(len(ratings_raw)),
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "duplicates_collapsed": int(movies_raw["movie_id"].nunique() - len(movies)),
        "tmdb_enrichment": "skipped_no_tmdb_key" if not os.getenv("TMDB_API_KEY") else "placeholder_run_separately",
    }
    (output_path / "letterboxd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def choose_canonical_movies(movies_raw: pd.DataFrame) -> pd.DataFrame:
    ranked = movies_raw.copy()
    ranked["has_url"] = ranked["movie_url"].fillna("").astype(str).str.strip().ne("")
    ranked["has_year"] = ranked["parsed_year"].fillna("").astype(str).str.strip().ne("")
    ranked = ranked.sort_values(["canonical_key", "has_url", "has_year", "movie_id"], ascending=[True, False, False, True])
    return ranked.drop_duplicates("canonical_key", keep="first").sort_values("canonical_key").reset_index(drop=True)


def canonical_key(row: pd.Series) -> str:
    slug = slug_from_url(row.get("movie_url"))
    if slug:
        return f"slug:{slug}"
    year = str(row.get("parsed_year") or "").strip()
    return f"title:{normalise_title(str(row.get('clean_title', '')))}:{year}"


def slug_from_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    parsed = urlparse(text)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "film":
        return normalise_title(parts[1])
    return ""


def parsed_year(year: Any, title: Any) -> str:
    try:
        if pd.notna(year) and str(year).strip():
            return str(int(float(str(year).strip())))
    except (TypeError, ValueError):
        pass
    match = re.search(r"\((\d{4})\)\s*$", str(title or ""))
    return match.group(1) if match else ""


def clean_title(title: Any) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title or "").strip())


def title_with_year(title: str, year: Any) -> str:
    year_text = str(year or "").strip()
    if year_text:
        return f"{title} ({year_text})"
    return title


def normalise_title(title: str) -> str:
    text = title.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def ordered_id_map(values: pd.Series, start: int = 1) -> dict[Any, int]:
    unique = values.dropna().drop_duplicates().tolist()
    return {value: idx + start for idx, value in enumerate(unique)}


def normalise_ratings(frame: pd.DataFrame) -> pd.Series:
    ratings = pd.to_numeric(frame["rating"], errors="coerce")
    if "liked" in frame.columns:
        liked = pd.to_numeric(frame["liked"], errors="coerce").fillna(0).gt(0)
        ratings = ratings.fillna(liked.map({True: 5.0, False: 0.0}))
    return ratings.fillna(0.0).clip(lower=0.5, upper=5.0).astype(float)


def stable_timestamps(frame: pd.DataFrame) -> pd.Series:
    if "watched_date" in frame.columns:
        parsed = pd.to_datetime(frame["watched_date"], errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.5:
            sequence = pd.Series(range(len(frame)), index=frame.index, dtype="int64")
            seconds = parsed.astype("int64") // 10**9
            return seconds.where(parsed.notna(), sequence).astype(int)
    return pd.Series(range(len(frame)), index=frame.index, dtype="int64")


def write_placeholder_enriched(path: Path, movies: pd.DataFrame) -> None:
    enriched = pd.DataFrame({"movieId": movies["movieId"].astype(int)})
    for column in PLACEHOLDER_METADATA_COLUMNS:
        if column == "movieId":
            continue
        if column in {"budget", "revenue", "runtime", "vote_average", "vote_count", "popularity", "collection_id"}:
            enriched[column] = 0
        elif column == "genres":
            enriched[column] = ""
        elif column == "enrichment_status":
            enriched[column] = "missing_enrichment_placeholder"
        else:
            enriched[column] = ""
    enriched[PLACEHOLDER_METADATA_COLUMNS].to_csv(path, index=False)


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Letterboxd crawl data as a MovieLens-style dataset.")
    parser.add_argument("--raw-dir", default="crawl/data/raw")
    parser.add_argument("--output-dir", default="data/letterboxd-full")
    parser.add_argument("--source", choices=["full", "cf"], default="full")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare_letterboxd(**vars(parse_args())), indent=2))


if __name__ == "__main__":
    main()
