from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app


class ApiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_recommend(self) -> None:
        response = self.client.post(
            "/recommend",
            json={"user_id": 104, "top_k": 3, "session_context": ["tmdb_862"]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("recommendations", payload)
        self.assertEqual(len(payload["recommendations"]), 3)
        self.assertIn("title", payload["recommendations"][0])

    def test_model_info(self) -> None:
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("model_info", payload)
        self.assertIn("model_source", payload["model_info"])
        self.assertIn("weights", payload["model_info"])


if __name__ == "__main__":
    unittest.main()
