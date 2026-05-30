from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory

from data import MovieLensDataLoader
from models import HybridMovieRecommender


class RecommenderSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = MovieLensDataLoader("data/sample").load()
        cls.model = HybridMovieRecommender().fit(bundle.movies, bundle.ratings, bundle.tags)

    def test_known_user_recommendations(self) -> None:
        recs = self.model.recommend(user_id=104, top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertTrue(all("score" in rec for rec in recs))

    def test_guest_session_recommendations(self) -> None:
        recs = self.model.recommend(user_id=None, session_context=["tmdb_862"], top_k=5)
        self.assertEqual(len(recs), 5)
        self.assertNotEqual(recs[0]["movie_id"], 1)

    def test_movie_picker_shape(self) -> None:
        movies = self.model.movies_for_picker()
        self.assertGreaterEqual(len(movies), 20)
        self.assertIn("title", movies[0])

    def test_predict_rating_range(self) -> None:
        rating = self.model.predict_rating(user_id=104, movie_id=21)
        self.assertGreaterEqual(rating, 0.5)
        self.assertLessEqual(rating, 5.0)

    def test_artifact_round_trip(self) -> None:
        bundle = MovieLensDataLoader("data/sample").load()
        with TemporaryDirectory() as temp_dir:
            self.model.save_artifact(
                temp_dir,
                dataset_name="sample",
                model_name="hybrid-test",
                metrics={"test": {"ndcg@10": 0.1}},
            )
            loaded = HybridMovieRecommender.from_artifact(temp_dir, bundle.movies, bundle.ratings, bundle.tags)
            recs = loaded.recommend(user_id=104, top_k=5)
            self.assertEqual(len(recs), 5)
            info = loaded.model_info()
            self.assertEqual(info["model_source"], "artifact")
            self.assertEqual(info["model_name"], "hybrid-test")

    def test_artifact_unknown_user_falls_back(self) -> None:
        bundle = MovieLensDataLoader("data/sample").load()
        with TemporaryDirectory() as temp_dir:
            self.model.save_artifact(temp_dir, dataset_name="sample")
            loaded = HybridMovieRecommender.from_artifact(temp_dir, bundle.movies, bundle.ratings, bundle.tags)
            recs = loaded.recommend(user_id=999999, top_k=3)
            self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()
