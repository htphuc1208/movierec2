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


if __name__ == "__main__":
    unittest.main()
