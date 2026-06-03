from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data import MovieLensDataLoader
from scripts.prepare_tag_genome import prepare_tag_genome


class MovieLensDataLoaderTest(unittest.TestCase):
    def test_enriched_genres_and_release_date_fill_missing_movie_fields(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            pd.DataFrame(
                {
                    "movieId": [1, 2],
                    "title": ["Range Show (2006-2007)", "Untitled Movie"],
                    "genres": ["Drama", "(no genres listed)"],
                }
            ).to_csv(path / "movies.csv", index=False)
            pd.DataFrame(
                {
                    "userId": [1, 1, 1],
                    "movieId": [1, 2, 1],
                    "rating": [4.0, 5.0, 3.0],
                    "timestamp": [1, 2, 3],
                }
            ).to_csv(path / "ratings.csv", index=False)
            pd.DataFrame({"movieId": [1, 2], "imdbId": ["tt1", "tt2"], "tmdbId": [10, 20]}).to_csv(
                path / "links.csv", index=False
            )
            pd.DataFrame(
                {
                    "movieId": [2],
                    "genres": ["Action|Sci-Fi"],
                    "release_date": ["2018-03-29"],
                    "overview": ["x"],
                    "production_companies": ["Warner Bros."],
                }
            ).to_csv(path / "enriched_movies.csv", index=False)

            movies = MovieLensDataLoader(path).load_movies()
            by_id = movies.set_index("movieId")

            self.assertEqual(str(by_id.loc[1, "year"]), "2006")
            self.assertEqual(by_id.loc[2, "genres"], "Action|Sci-Fi")
            self.assertEqual(str(by_id.loc[2, "year"]), "2018")
            self.assertEqual(by_id.loc[2, "production_companies"], "Warner Bros.")

    def test_warm_cold_split_and_tag_genome_features(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)
            genome_dir = path / "genome"
            genome_dir.mkdir()
            pd.DataFrame(
                {
                    "movieId": [1, 2, 3],
                    "title": ["A (2000)", "B (2001)", "C (2002)"],
                    "genres": ["Drama", "Comedy", "Action"],
                }
            ).to_csv(path / "movies.csv", index=False)
            ratings = pd.DataFrame(
                {
                    "userId": [1, 1, 1, 2],
                    "movieId": [1, 2, 3, 1],
                    "rating": [5.0, 4.0, 5.0, 3.0],
                    "timestamp": [1, 2, 3, 1],
                }
            )
            ratings.to_csv(path / "ratings.csv", index=False)
            pd.DataFrame({"movieId": [1, 2], "tagId": [10, 11], "relevance": [0.9, 0.8]}).to_csv(
                genome_dir / "genome-scores.csv",
                index=False,
            )
            pd.DataFrame({"tagId": [10, 11], "tag": ["mind-bending", "witty"]}).to_csv(
                genome_dir / "genome-tags.csv",
                index=False,
            )

            prepare_tag_genome(path, genome_dir, top_n=5, min_relevance=0.1)
            movies = MovieLensDataLoader(path).load_movies()
            by_id = movies.set_index("movieId")
            self.assertEqual(by_id.loc[1, "tag_genome_tags"], "mind-bending")

            train = ratings.iloc[:2]
            holdout = ratings.iloc[2:]
            warm, cold = MovieLensDataLoader.split_warm_cold_items(train, holdout)
            self.assertEqual(warm["movieId"].astype(int).tolist(), [1])
            self.assertEqual(cold["movieId"].astype(int).tolist(), [3])


if __name__ == "__main__":
    unittest.main()
