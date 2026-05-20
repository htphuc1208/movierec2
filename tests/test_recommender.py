from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
