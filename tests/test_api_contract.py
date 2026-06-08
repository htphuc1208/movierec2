from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from recommender.inference.artifacts import save_artifact_bundle


pytestmark = pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi is not installed")


def test_api_health_and_recommendations(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from api.main import app, get_chatbot, get_recommender

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
        hybrid_config={"content_weight": 1.0, "popularity_weight": 0.0, "cf_weight": 0.0, "train_user_items": {"0": [0]}},
    )
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("RATINGS_STORE_PATH", str(tmp_path / "runtime" / "ratings.csv"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_recommender.cache_clear()
    from api.main import get_rating_store

    get_rating_store.cache_clear()
    get_chatbot.cache_clear()

    client = TestClient(app)
    assert client.get("/health").json()["artifacts"]["ready"]
    assert client.get("/users").json()["users"] == [1]
    response = client.post("/recommendations", json={"user_id": 1, "top_k": 1, "session_context": []})
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) == 1
    alias = client.post("/recommend", json={"user_id": 1, "top_k": 1, "session_context": [], "model_name": "SVD"})
    assert alias.status_code == 200

    detail = client.get("/movies/2")
    assert detail.status_code == 200
    assert detail.json()["title"] == "B"
    assert client.get("/movies/2/similar").status_code == 200
    assert client.get("/movies/trending").status_code == 200
    assert client.get("/movies/top-rated").status_code == 200
    assert client.get("/movies/latest").status_code == 200
    assert client.get("/movies/genre/Drama").status_code == 200
    assert client.get("/model-info").json()["model_info"]["movie_count"] == 2

    saved = client.post("/rate", json={"user_id": 1, "movie_id": 2, "rating": 4.5})
    assert saved.status_code == 200
    assert client.get("/rate/1/2").json()["rating"] == 4.5
    history = client.get("/users/1/history").json()["movies"]
    assert {movie["movie_id"] for movie in history} == {1, 2}
    chat = client.post("/chat", json={"message": "phim drama", "top_k": 1})
    assert chat.status_code == 200
    assert chat.json()["sources"]
