from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.reprocess_crawl_movielens import reprocess


class ReprocessCrawlMovieLensTest(unittest.TestCase):
    def test_reprocess_outputs_movielens_style_files(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            output_dir = root / "out"
            raw_dir.mkdir()

            pd.DataFrame(
                {
                    "movie_id": ["tt1", "tt2"],
                    "title": ["Alpha", "Beta"],
                    "year": [2001, 2002],
                }
            ).to_csv(raw_dir / "movies_cf.csv", index=False)
            pd.DataFrame(
                {
                    "user_id": ["u1", "u1", "u2"],
                    "movie_id": ["tt1", "tt1", "tt2"],
                    "rating": [3.0, 4.0, 5.0],
                    "watched_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                }
            ).to_csv(raw_dir / "ratings_cf.csv", index=False)

            summary = reprocess(raw_dir, output_dir)

            movies = pd.read_csv(output_dir / "movies.csv")
            ratings = pd.read_csv(output_dir / "ratings.csv")
            tags = pd.read_csv(output_dir / "tags.csv")
            movie_map = pd.read_csv(output_dir / "movie_id_mapping.csv")

            self.assertEqual(summary["movies"], 2)
            self.assertEqual(summary["ratings"], 2)
            self.assertEqual(movies["title"].tolist(), ["Alpha (2001)", "Beta (2002)"])
            self.assertEqual(ratings["rating"].tolist(), [4.0, 5.0])
            self.assertEqual(tags.columns.tolist(), ["userId", "movieId", "tag", "timestamp"])
            self.assertEqual(movie_map["movieId"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
