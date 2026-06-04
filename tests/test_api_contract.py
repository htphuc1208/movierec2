from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from recommender.inference.artifacts import save_artifact_bundle


pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi is not installed")


def test_api_health_and_recommendations(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from api.main import app, get_recommender

    save_artifact_bundle(
        tmp_path,
        catalog=pd.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["A", "B"],
                "genres": ["Drama", "Drama"],
                "tmdb_id": [10, 11],
                "poster_url": [None, None],
            }
        ),
        user_mapping={1: 0},
        item_mapping={1: 0, 2: 1},
        content_embeddings=np.array([[1, 0], [0.9, 0.1]], dtype=np.float32),
        user_profiles=np.array([[1, 0]], dtype=np.float32),
        item_popularity=np.array([0.1, 0.2], dtype=np.float32),
        metrics={},
        hybrid_config={"content_weight": 1.0, "popularity_weight": 0.0, "cf_weight": 0.0},
    )
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    get_recommender.cache_clear()

    client = TestClient(app)
    assert client.get("/health").json()["artifacts"]["ready"]
    response = client.post("/recommendations", json={"user_id": 1, "top_k": 1, "session_context": []})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 1
