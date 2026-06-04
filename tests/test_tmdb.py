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
            "vote_count": 123,
            "runtime": 81,
            "original_language": "en",
            "production_countries": [{"name": "United States of America"}],
            "production_companies": [{"name": "Pixar"}],
            "belongs_to_collection": {"name": "Toy Story Collection"},
        }

    def movie_details(self, tmdb_id: int):
        payload = self.movie(tmdb_id)
        payload["credits"] = self.credits(tmdb_id)
        payload["keywords"] = self.keywords(tmdb_id)
        return payload

    def credits(self, tmdb_id: int):
        return {
            "crew": [
                {"job": "Director", "name": "Jane Director"},
                {"job": "Screenplay", "name": "Jane Writer"},
            ],
            "cast": [{"order": 0, "name": "Lead Actor"}],
        }

    def keywords(self, tmdb_id: int):
        return {"keywords": [{"name": "friendship"}, {"name": "toy"}]}


def test_normalize_tmdb_payload() -> None:
    payload = normalize_tmdb_payload(
        {
            "id": 1,
            "poster_path": "/a.jpg",
            "release_date": "2000-01-01",
            "genres": [{"name": "Drama"}],
            "belongs_to_collection": {"name": "Series"},
        },
        {
            "crew": [{"job": "Director", "name": "Director"}, {"job": "Writer", "name": "Writer"}],
            "cast": [{"order": 0, "name": "Actor"}],
        },
        {"keywords": [{"name": "memory"}]},
    )
    assert payload["poster_url"].endswith("/w500/a.jpg")
    assert "tagline" not in payload
    assert payload["release_year"] == "2000"
    assert payload["keywords"] == "memory"
    assert payload["collection"] == "Series"
    assert payload["director"] == "Director"
    assert payload["writers"] == "Writer"
    assert payload["cast"] == "Actor"


def test_enrich_catalog_uses_cache(tmp_path) -> None:
    movies = pd.DataFrame({"movieId": [1], "title": ["Toy Story (1995)"], "genres": ["Animation"]})
    links = pd.DataFrame({"movieId": [1], "imdbId": [114709], "tmdbId": [862]})

    enriched = enrich_catalog(movies, links, FakeTMDBClient(), tmp_path / "cache.json", sleep_seconds=0)

    assert enriched.iloc[0]["tmdb_id"] == 862
    assert enriched.iloc[0]["director"] == "Jane Director"
    assert enriched.iloc[0]["writers"] == "Jane Writer"
    assert enriched.iloc[0]["keywords"] == "friendship|toy"
    assert enriched.iloc[0]["runtime_minutes"] == 81
    assert enriched.iloc[0]["collection"] == "Toy Story Collection"
    assert "tagline" not in enriched.columns
    assert (tmp_path / "cache.json").exists()
