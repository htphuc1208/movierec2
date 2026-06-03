from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.prepare_letterboxd import prepare_letterboxd


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


if __name__ == "__main__":
    unittest.main()
