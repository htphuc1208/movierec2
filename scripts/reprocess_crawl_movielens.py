from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def read_first_existing(raw_dir: Path, names: list[str]) -> pd.DataFrame:
    for name in names:
        path = raw_dir / name
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of these files exist in {raw_dir}: {', '.join(names)}")


def ordered_id_map(values: pd.Series, start: int = 1) -> dict[Any, int]:
    unique = values.dropna().drop_duplicates().tolist()
    return {value: idx + start for idx, value in enumerate(unique)}


def parse_timestamp(frame: pd.DataFrame) -> pd.Series:
    for column in ["watched_date", "created_at"]:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if parsed.notna().any():
                return parsed.fillna(pd.Timestamp.now(tz="UTC")).astype("int64") // 10**9
    return pd.Series(range(len(frame)), index=frame.index, dtype="int64")


def normalise_ratings(frame: pd.DataFrame) -> pd.Series:
    if "rating" in frame.columns:
        ratings = pd.to_numeric(frame["rating"], errors="coerce")
    elif "implicit_score" in frame.columns:
        ratings = pd.to_numeric(frame["implicit_score"], errors="coerce")
    else:
        ratings = pd.Series(0.0, index=frame.index)

    if "liked" in frame.columns:
        liked = frame["liked"].fillna("").astype(str).str.lower().isin(["1", "true", "yes", "liked"])
        ratings = ratings.fillna(liked.map({True: 5.0, False: 0.0}))
    ratings = ratings.fillna(0.0).clip(lower=0.5, upper=5.0)
    return ratings.astype(float)


def title_with_year(title: Any, year: Any) -> str:
    title_text = str(title or "").strip()
    year_text = str(year or "").strip()
    if year_text and year_text.endswith(".0"):
        year_text = year_text[:-2]
    if year_text and f"({year_text})" not in title_text:
        return f"{title_text} ({year_text})"
    return title_text


def reprocess(raw_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movies_raw = read_first_existing(raw_path, ["movies_cf.csv", "movies_seed.csv"])
    ratings_raw = read_first_existing(raw_path, ["ratings_cf.csv", "interactions.csv"])

    required_movie_cols = {"movie_id", "title"}
    required_rating_cols = {"user_id", "movie_id"}
    if not required_movie_cols.issubset(movies_raw.columns):
        raise ValueError(f"Movie crawl data must contain: {sorted(required_movie_cols)}")
    if not required_rating_cols.issubset(ratings_raw.columns):
        raise ValueError(f"Rating crawl data must contain: {sorted(required_rating_cols)}")

    movies_raw = movies_raw.dropna(subset=["movie_id", "title"]).drop_duplicates("movie_id").copy()
    ratings_raw = ratings_raw.dropna(subset=["user_id", "movie_id"]).copy()
    ratings_raw = ratings_raw.loc[ratings_raw["movie_id"].isin(set(movies_raw["movie_id"]))].copy()

    movie_map = ordered_id_map(movies_raw["movie_id"])
    user_map = ordered_id_map(ratings_raw["user_id"])

    movies = pd.DataFrame(
        {
            "movieId": movies_raw["movie_id"].map(movie_map).astype(int),
            "title": [
                title_with_year(row.title, getattr(row, "year", ""))
                for row in movies_raw.itertuples()
            ],
            "genres": "(no genres listed)",
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": ratings_raw["user_id"].map(user_map).astype(int),
            "movieId": ratings_raw["movie_id"].map(movie_map).astype(int),
            "rating": normalise_ratings(ratings_raw),
            "timestamp": parse_timestamp(ratings_raw).astype(int),
        }
    )
    ratings = ratings.drop_duplicates(["userId", "movieId"], keep="last").sort_values(["userId", "timestamp", "movieId"])

    movie_mapping = pd.DataFrame(
        [{"raw_movie_id": raw_id, "movieId": movie_id} for raw_id, movie_id in movie_map.items()]
    )
    user_mapping = pd.DataFrame(
        [{"raw_user_id": raw_id, "userId": user_id} for raw_id, user_id in user_map.items()]
    )
    tags = pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    movies.to_csv(output_path / "movies.csv", index=False)
    ratings.to_csv(output_path / "ratings.csv", index=False)
    tags.to_csv(output_path / "tags.csv", index=False)
    movie_mapping.to_csv(output_path / "movie_id_mapping.csv", index=False)
    user_mapping.to_csv(output_path / "user_id_mapping.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_path),
        "output_dir": str(output_path),
        "movies": int(len(movies)),
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
    }
    (output_path / "reprocess_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert crawled movie/user ids into MovieLens-style CSV files.")
    parser.add_argument("--raw-dir", default="crawl/data/raw")
    parser.add_argument("--output-dir", default="crawl/data/reprocessing")
    return parser.parse_args()


def main() -> None:
    summary = reprocess(**vars(parse_args()))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
