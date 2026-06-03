from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

OPTIONAL_TEXT_METADATA_COLUMNS = [
    "keywords",
    "tag_genome_tags",
    "original_language",
    "production_companies",
    "production_countries",
    "collection_name",
    "release_date",
    "certification",
]

OPTIONAL_NUMERIC_METADATA_COLUMNS = [
    "runtime",
    "vote_average",
    "vote_count",
    "popularity",
    "collection_id",
]


@dataclass(frozen=True)
class MovieDataBundle:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


class MovieLensDataLoader:
    """Load MovieLens-style CSV files and optional enriched metadata."""

    def __init__(self, data_dir: str | Path = "data/sample") -> None:
        self.data_dir = Path(data_dir)

    def load(self) -> MovieDataBundle:
        return MovieDataBundle(
            movies=self.load_movies(),
            ratings=self.load_ratings(),
            tags=self.load_tags(),
            links=self.load_links(),
        )

    def load_movies(self) -> pd.DataFrame:
        movies = self._read_required_csv("movies.csv", dtype={"movieId": "int32"})
        movies["movieId"] = movies["movieId"].astype(int)
        movies["title"] = movies["title"].astype(str)
        movies["genres"] = movies["genres"].fillna("").astype(str)

        if "year" not in movies.columns:
            movies["year"] = self._extract_year(movies["title"])

        links_path = self.data_dir / "links.csv"
        if links_path.exists():
            links = pd.read_csv(links_path, dtype={"movieId": "int32"})
            links["movieId"] = links["movieId"].astype(int)
            movies = movies.merge(links, on="movieId", how="left")

        enriched_path = self.data_dir / "enriched_movies.csv"
        if enriched_path.exists():
            enriched = pd.read_csv(enriched_path)
            enriched["movieId"] = enriched["movieId"].astype(int)
            movies = movies.merge(enriched, on="movieId", how="left", suffixes=("", "_enriched"))
            for column in ["overview", "tagline", "director", "cast", "poster_url", "budget", "revenue"]:
                enriched_column = f"{column}_enriched"
                if enriched_column in movies.columns:
                    if column not in movies.columns:
                        movies[column] = pd.NA
                    movies[column] = movies[column].combine_first(movies[enriched_column])
                    movies = movies.drop(columns=[enriched_column])
            if "directors" in movies.columns:
                if "director" not in movies.columns:
                    movies["director"] = pd.NA
                movies["director"] = movies["director"].replace("", pd.NA).combine_first(movies["directors"])
                movies = movies.drop(columns=["directors"])
            if "top_cast" in movies.columns:
                if "cast" not in movies.columns:
                    movies["cast"] = pd.NA
                movies["cast"] = movies["cast"].replace("", pd.NA).combine_first(movies["top_cast"])
                movies = movies.drop(columns=["top_cast"])
            if "tmdb_poster_path" in movies.columns:
                if "poster_url" not in movies.columns:
                    movies["poster_url"] = pd.NA
                poster_path = movies["tmdb_poster_path"].fillna("").astype(str).str.strip()
                poster_url = poster_path.where(poster_path.eq("") | poster_path.str.startswith("http"), "https://image.tmdb.org/t/p/w500" + poster_path)
                movies["poster_url"] = movies["poster_url"].replace("", pd.NA).combine_first(poster_url).fillna("")
                movies = movies.drop(columns=["tmdb_poster_path"])
            if "genres_enriched" in movies.columns:
                missing_genres = movies["genres"].fillna("").astype(str).str.strip().isin(["", "(no genres listed)"])
                enriched_genres = movies["genres_enriched"].fillna("").astype(str).str.strip()
                movies.loc[missing_genres & enriched_genres.ne(""), "genres"] = enriched_genres
                movies = movies.drop(columns=["genres_enriched"])
            if "year_enriched" in movies.columns:
                movies["year"] = movies["year"].replace("", pd.NA).combine_first(movies["year_enriched"])
                movies = movies.drop(columns=["year_enriched"])
            if "release_date" in movies.columns:
                release_year = movies["release_date"].fillna("").astype(str).str.extract(r"^(\d{4})")[0].fillna("")
                movies["year"] = movies["year"].replace("", pd.NA).combine_first(release_year).fillna("")

        movies = self._merge_tag_genome(movies)

        for column in ["overview", "tagline", "director", "cast", "poster_url"]:
            if column not in movies.columns:
                movies[column] = ""
            movies[column] = movies[column].fillna("").astype(str)

        for column in ["tmdbId", "imdbId"]:
            if column not in movies.columns:
                movies[column] = ""
            movies[column] = movies[column].fillna("")

        for column in OPTIONAL_TEXT_METADATA_COLUMNS:
            if column not in movies.columns:
                movies[column] = ""
            movies[column] = movies[column].fillna("").astype(str)

        for column in OPTIONAL_NUMERIC_METADATA_COLUMNS:
            if column not in movies.columns:
                movies[column] = 0
            movies[column] = pd.to_numeric(movies[column], errors="coerce").fillna(0)

        return movies

    def load_ratings(self) -> pd.DataFrame:
        ratings = self._read_required_csv(
            "ratings.csv",
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "rating": "float32",
                "timestamp": "int64",
            },
        )
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype("float32")
        if "timestamp" not in ratings.columns:
            ratings["timestamp"] = range(len(ratings))
        return ratings

    def load_tags(self) -> pd.DataFrame:
        path = self.data_dir / "tags.csv"
        if not path.exists():
            return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        tags = pd.read_csv(path, dtype={"userId": "int32", "movieId": "int32"})
        if tags.empty:
            return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        tags["movieId"] = tags["movieId"].astype(int)
        tags["tag"] = tags["tag"].fillna("").astype(str)
        return tags

    def load_links(self) -> pd.DataFrame:
        path = self.data_dir / "links.csv"
        if not path.exists():
            return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
        links = pd.read_csv(path, dtype={"movieId": "int32"})
        links["movieId"] = links["movieId"].astype(int)
        return links

    def train_val_test_split(
        self,
        ratings: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split each user's interactions by timestamp to avoid future leakage."""

        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for _, user_rows in ratings.sort_values("timestamp").groupby("userId"):
            count = len(user_rows)
            if count < 3:
                train_parts.append(user_rows)
                continue
            test_size = max(1, int(round(count * test_ratio)))
            val_size = max(1, int(round(count * val_ratio)))
            train_end = max(1, count - val_size - test_size)
            val_end = count - test_size
            train_parts.append(user_rows.iloc[:train_end])
            val_parts.append(user_rows.iloc[train_end:val_end])
            test_parts.append(user_rows.iloc[val_end:])

        return (
            self._concat_or_empty(train_parts, ratings.columns),
            self._concat_or_empty(val_parts, ratings.columns),
            self._concat_or_empty(test_parts, ratings.columns),
        )

    @staticmethod
    def rated_movies(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
        """Return catalog rows with at least one rating in the provided ratings frame."""

        rated_movie_ids = set(ratings["movieId"].astype(int).unique().tolist())
        return movies.loc[movies["movieId"].astype(int).isin(rated_movie_ids)].reset_index(drop=True)

    @staticmethod
    def split_warm_cold_items(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split holdout interactions by whether the item appears in train."""

        train_movie_ids = set(train["movieId"].astype(int).unique().tolist())
        warm_mask = holdout["movieId"].astype(int).isin(train_movie_ids)
        return holdout.loc[warm_mask].copy(), holdout.loc[~warm_mask].copy()

    @staticmethod
    def build_implicit_interactions(
        ratings: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> pd.DataFrame:
        interactions = ratings.loc[ratings["rating"] >= positive_threshold, ["userId", "movieId", "rating"]]
        return interactions.drop_duplicates(["userId", "movieId"]).reset_index(drop=True)

    @staticmethod
    def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{source} is missing required columns: {joined}")

    def _read_required_csv(self, filename: str, **kwargs: object) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
        return pd.read_csv(path, **kwargs)

    def _merge_tag_genome(self, movies: pd.DataFrame) -> pd.DataFrame:
        tag_genome_path = self.data_dir / "tag_genome.csv"
        if not tag_genome_path.exists():
            return movies

        tag_genome = pd.read_csv(tag_genome_path)
        if tag_genome.empty or "movieId" not in tag_genome.columns:
            return movies

        tag_genome["movieId"] = tag_genome["movieId"].astype(int)
        if {"tag", "relevance"}.issubset(tag_genome.columns):
            tag_genome["tag"] = tag_genome["tag"].fillna("").astype(str)
            tag_genome["relevance"] = pd.to_numeric(tag_genome["relevance"], errors="coerce").fillna(0)
            top_tags = (
                tag_genome.sort_values(["movieId", "relevance"], ascending=[True, False])
                .groupby("movieId")
                .head(20)
                .groupby("movieId")["tag"]
                .apply(lambda values: " ".join(value for value in values if value.strip()))
                .reset_index(name="tag_genome_tags")
            )
        elif "tag_genome_tags" in tag_genome.columns:
            top_tags = tag_genome[["movieId", "tag_genome_tags"]].copy()
        else:
            return movies

        movies = movies.merge(top_tags, on="movieId", how="left", suffixes=("", "_tag_genome"))
        if "tag_genome_tags_tag_genome" in movies.columns:
            movies["tag_genome_tags"] = movies.get("tag_genome_tags", pd.Series("", index=movies.index)).replace("", pd.NA).combine_first(movies["tag_genome_tags_tag_genome"])
            movies = movies.drop(columns=["tag_genome_tags_tag_genome"])
        return movies

    @staticmethod
    def _concat_or_empty(parts: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _extract_year(titles: pd.Series) -> pd.Series:
        return titles.astype(str).str.extract(r"\((\d{4})(?:[–-]\d{4})?\)")[0].fillna("")
