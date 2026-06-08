from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.prepare_letterboxd import prepare_letterboxd
from scripts.enrich_tmdb import movie_query_and_year, resolve_missing_tmdb_ids


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls = []

    def get(self, url, params=None, timeout=None):  # noqa: ANN001
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse({"results": [{"id": 12345}]})


class PrepareLetterboxdTest(unittest.TestCase):
    def test_prepare_full_letterboxd_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            output_dir = root / "letterboxd-full"
            raw_dir.mkdir()
            pd.DataFrame(
                {
                    "movie_id": ["raw-a", "raw-a-dup", "raw-b"],
                    "title": ["The Alpha (2001)", "Alpha (2001)", "Beta"],
                    "year": [2001, 2001, ""],
                    "movie_url": [
                        "https://letterboxd.com/film/alpha/",
                        "https://letterboxd.com/film/alpha/",
                        "",
                    ],
                }
            ).to_csv(raw_dir / "movies_seed.csv", index=False)
            pd.DataFrame(
                {
                    "user_id": ["u1", "u1", "u2"],
                    "movie_id": ["raw-a", "raw-a-dup", "raw-b"],
                    "rating": [4.0, 5.0, 0.0],
                    "liked": ["", "", ""],
                    "watched_date": ["", "", ""],
                }
            ).to_csv(raw_dir / "ratings.csv", index=False)

            summary = prepare_letterboxd(raw_dir, output_dir, source="full")
            movies = pd.read_csv(output_dir / "movies.csv")
            ratings = pd.read_csv(output_dir / "ratings.csv")
            links = pd.read_csv(output_dir / "links.csv")
            enriched = pd.read_csv(output_dir / "enriched_movies.csv")
            movie_mapping = pd.read_csv(output_dir / "movie_id_mapping.csv")

            self.assertEqual(summary["movies"], 2)
            self.assertEqual(len(movies), 2)
            self.assertEqual(len(links), 2)
            self.assertFalse(ratings.duplicated(["userId", "movieId"]).any())
            self.assertTrue(ratings["rating"].between(0.5, 5.0).all())
            self.assertEqual(enriched["enrichment_status"].unique().tolist(), ["missing_enrichment_placeholder"])
            self.assertEqual(movie_mapping.loc[movie_mapping["raw_movie_id"] == "raw-a", "movieId"].iloc[0], 1)
            self.assertEqual(movie_mapping.loc[movie_mapping["raw_movie_id"] == "raw-a-dup", "movieId"].iloc[0], 1)

    def test_tmdb_search_resolver_uses_title_and_year(self) -> None:
        movies = pd.DataFrame({"movieId": [1], "title": ["Alpha (2001)"], "year": ["2001"]})
        links = pd.DataFrame({"movieId": [1], "imdbId": [""], "tmdbId": [""]})
        selected, updated_links, summary = resolve_missing_tmdb_ids(
            links,
            links,
            movies,
            "fake-key",
            FakeSession(),
            sleep=0.0,
        )
        self.assertEqual(summary, {"searched": 1, "matched": 1})
        self.assertEqual(str(selected.loc[0, "tmdbId"]), "12345")
        self.assertEqual(str(updated_links.loc[0, "tmdbId"]), "12345")
        self.assertEqual(movie_query_and_year({"title": "Beta (1999)", "year": ""}), ("Beta", "1999"))


if __name__ == "__main__":
    unittest.main()
