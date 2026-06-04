from __future__ import annotations

import pandas as pd

from recommender.data.letterboxd import (
    build_base_letterboxd_catalog,
    enrich_letterboxd_catalog,
    materialize_letterboxd,
)


class FakeTMDBClient:
    def search_movie(self, query: str, year=None, include_adult: bool = False):
        return {
            "results": [
                {
                    "id": 99,
                    "title": query,
                    "original_title": query,
                    "release_date": f"{year or 2001}-01-01",
                }
            ]
        }

    def movie_details(self, tmdb_id: int):
        return {
            "id": tmdb_id,
            "title": "Movie A",
            "overview": "A test movie.",
            "release_date": "2001-01-01",
            "genres": [{"name": "Drama"}],
            "poster_path": "/poster.jpg",
            "credits": {
                "crew": [{"job": "Director", "name": "Director A"}, {"job": "Writer", "name": "Writer A"}],
                "cast": [{"order": 0, "name": "Actor A"}],
            },
            "keywords": {"keywords": [{"name": "test"}]},
        }


def write_letterboxd_raw(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2"],
            "movie_id": ["m1", "m2", "m3", "m1"],
            "interaction_type": ["rating", "watched", "favorite", "rating"],
            "rating": [4.5, None, 5.0, 3.0],
            "implicit_score": [4.5, 2.5, 5.0, 3.0],
            "source": ["films_page", "films_page", "profile_favorites", "films_page"],
            "watched_date": ["Fri, 1 Aug 2025 01:32:50 +1200", None, "bad-date", None],
            "created_at": ["2026-01-01T00:00:00+00:00"] * 4,
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": ["m1", "m2", "m3"],
            "title": ["Movie A (2001)", "Movie B", "Movie C"],
            "year": [None, 2002, None],
            "movie_url": ["https://letterboxd.com/film/movie-a/", "", ""],
            "created_at": ["2026-01-01T00:00:00+00:00"] * 3,
        }
    )
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "username": ["one", "two"],
            "display_name": ["One", "Two"],
            "profile_url": ["https://letterboxd.com/one/", "https://letterboxd.com/two/"],
            "films_count": [10, 20],
            "following_count": [1, 2],
            "followers_count": [3, 4],
            "created_at": ["2026-01-01T00:00:00+00:00"] * 2,
        }
    )
    interactions.to_csv(raw / "interactions_cf.csv", index=False)
    movies.to_csv(raw / "movies_cf.csv", index=False)
    users.to_csv(raw / "users.csv", index=False)


def test_materialize_letterboxd_implicit_uses_synthetic_timestamp(tmp_path) -> None:
    write_letterboxd_raw(tmp_path)
    out = tmp_path / "out"

    prepared = materialize_letterboxd(tmp_path / "raw", out, split="cf", rating_policy="implicit", seed=7)

    assert (out / "ratings.csv").exists()
    assert (out / "movies.csv").exists()
    assert len(prepared.ratings) == 4
    assert set(prepared.interactions_debug["timestamp_source"]) == {"watched_date", "missing"}
    assert prepared.summary["split_strategy"] == "synthetic_random_per_user"
    assert "created_at is crawler time" in prepared.summary["note"]

    rerun = materialize_letterboxd(tmp_path / "raw", tmp_path / "out2", split="cf", rating_policy="implicit", seed=7)
    assert prepared.ratings["timestamp"].tolist() == rerun.ratings["timestamp"].tolist()


def test_materialize_letterboxd_explicit_filters_non_rating(tmp_path) -> None:
    write_letterboxd_raw(tmp_path)

    prepared = materialize_letterboxd(tmp_path / "raw", tmp_path / "out", split="cf", rating_policy="explicit")

    assert len(prepared.ratings) == 2
    assert prepared.interactions_debug["interaction_type"].tolist() == ["rating", "rating"]


def test_base_and_tmdb_enriched_catalog(tmp_path) -> None:
    write_letterboxd_raw(tmp_path)
    out = tmp_path / "out"
    materialize_letterboxd(tmp_path / "raw", out, split="cf", rating_policy="implicit")

    base_catalog = build_base_letterboxd_catalog(out)
    assert "letterboxd_movie_id" in base_catalog.columns
    assert "tmdb_id" in base_catalog.columns

    enriched = enrich_letterboxd_catalog(out, FakeTMDBClient(), out / "cache.json", sleep_seconds=0)
    assert enriched.iloc[0]["tmdb_id"] == 99
    assert enriched.iloc[0]["tmdb_genres"] == "Drama"
    assert enriched.iloc[0]["director"] == "Director A"
