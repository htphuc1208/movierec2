from __future__ import annotations

import os
import unittest

from api import main as api_main


class ApiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MOVIEREC_DATA_DIR"] = "data/sample"
        os.environ.pop("MOVIEREC_ARTIFACT_DIR", None)
        api_main.get_recommender.cache_clear()

    def tearDown(self) -> None:
        api_main.get_recommender.cache_clear()

    def test_health(self) -> None:
        self.assertEqual(api_main.health(), {"status": "ok"})

    def test_recommend(self) -> None:
        response = api_main.recommend(
            api_main.RecommendRequest(user_id=104, top_k=3, session_context=["tmdb_862"])
        )
        self.assertEqual(len(response.recommendations), 3)
        self.assertIn("title", response.recommendations[0])

    def test_model_info(self) -> None:
        response = api_main.model_info()
        self.assertIn("model_source", response.model_info)
        self.assertIn("weights", response.model_info)

    def test_movie_catalog_views(self) -> None:
        self.assertGreater(len(api_main.movies(search="Toy")["movies"]), 0)
        self.assertGreater(len(api_main.trending_movies(top_k=3, min_votes=0)["movies"]), 0)
        self.assertGreater(len(api_main.top_rated_movies(top_k=3, min_votes=0)["movies"]), 0)
        self.assertGreater(len(api_main.latest_movies(top_k=3)["movies"]), 0)
        self.assertGreater(len(api_main.genre_movies("Adventure", top_k=3)["movies"]), 0)

    def test_movie_details_similar_and_history(self) -> None:
        detail = api_main.movie_details(1)
        self.assertEqual(detail["movieId"], 1)
        self.assertIn("rating_mean", detail)

        similar = api_main.similar_movies(1, top_k=3)["movies"]
        self.assertEqual(len(similar), 3)
        self.assertTrue(all(movie["movieId"] != 1 for movie in similar))

        history = api_main.user_history(104, top_k=3)["movies"]
        self.assertGreater(len(history), 0)
        self.assertIn("user_rating", history[0])


if __name__ == "__main__":
    unittest.main()
