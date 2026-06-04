from __future__ import annotations

import pandas as pd

from recommender.data.tmdb import enrich_catalog, normalize_tmdb_payload


class FakeTMDBClient:
    def movie(self, tmdb_id: int):
        return {
            "id": tmdb_id,
            "overview": "A space adventure.",
            "tagline": "To infinity.",
            "poster_path": "/poster.jpg",
            "release_date": "1995-11-22",
            "genres": [{"name": "Animation"}],
            "popularity": 10,
            "vote_average": 8.2,
        }

    def credits(self, tmdb_id: int):
        return {
            "crew": [{"job": "Director", "name": "Jane Director"}],
            "cast": [{"order": 0, "name": "Lead Actor"}],
        }


def test_normalize_tmdb_payload() -> None:
    payload = normalize_tmdb_payload(
        {"id": 1, "poster_path": "/a.jpg", "genres": [{"name": "Drama"}]},
        {"crew": [{"job": "Director", "name": "Director"}], "cast": [{"order": 0, "name": "Actor"}]},
    )
    assert payload["poster_url"].endswith("/w500/a.jpg")
    assert payload["director"] == "Director"
    assert payload["cast"] == "Actor"


def test_enrich_catalog_uses_cache(tmp_path) -> None:
    movies = pd.DataFrame({"movieId": [1], "title": ["Toy Story (1995)"], "genres": ["Animation"]})
    links = pd.DataFrame({"movieId": [1], "imdbId": [114709], "tmdbId": [862]})

    enriched = enrich_catalog(movies, links, FakeTMDBClient(), tmp_path / "cache.json", sleep_seconds=0)

    assert enriched.iloc[0]["tmdb_id"] == 862
    assert enriched.iloc[0]["director"] == "Jane Director"
    assert (tmp_path / "cache.json").exists()
