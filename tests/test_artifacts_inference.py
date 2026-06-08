from __future__ import annotations

import numpy as np
import pandas as pd

from recommender.inference.artifacts import artifact_status, save_artifact_bundle
from recommender.inference.ratings_store import SidecarRatingStore
from recommender.inference.recommender import HybridArtifactRecommender


def make_bundle(tmp_path):
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["Space One", "Space Two", "Kitchen Drama"],
            "genres": ["Sci-Fi", "Sci-Fi", "Drama"],
            "tmdb_id": [100, 101, 102],
            "poster_url": ["http://img/1.jpg", "http://img/2.jpg", "http://img/3.jpg"],
            "director": ["A", "B", "C"],
        }
    )
    content_embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    user_profiles = np.array([[1, 0]], dtype=np.float32)
    save_artifact_bundle(
        tmp_path,
        catalog=catalog,
        user_mapping={1: 0},
        item_mapping={10: 0, 11: 1, 12: 2},
        content_embeddings=content_embeddings,
        user_profiles=user_profiles,
        item_popularity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        metrics={"test": {"ndcg@10": 1.0}},
        hybrid_config={"content_weight": 0.9, "popularity_weight": 0.1, "cf_weight": 0.0, "train_user_items": {"0": [0]}},
    )


def test_artifact_status_and_recommend(tmp_path) -> None:
    make_bundle(tmp_path)
    assert artifact_status(tmp_path)["ready"]

    recommender = HybridArtifactRecommender.from_dir(tmp_path)
    results = recommender.recommend(user_id=1, top_k=2)

    assert len(results) == 2
    assert results[0]["movie_id"] in {10, 11}
    assert results[0]["poster_url"]


def test_search_and_session_context(tmp_path) -> None:
    make_bundle(tmp_path)
    recommender = HybridArtifactRecommender.from_dir(tmp_path)

    assert recommender.search_movies("space", limit=5)
    results = recommender.recommend(top_k=1, session_context=["tmdb_100"])
    assert results[0]["movie_id"] == 11


def test_catalog_helpers_and_rating_history(tmp_path) -> None:
    make_bundle(tmp_path)
    recommender = HybridArtifactRecommender.from_dir(tmp_path)
    store = SidecarRatingStore(tmp_path / "runtime" / "ratings.csv")
    store.append(user_id=1, movie_id=12, rating=4.5, timestamp=123)

    assert recommender.users() == [1]
    assert recommender.movie_detail(10)["title"] == "Space One"
    similar = recommender.similar_movies(10, top_k=1)
    assert similar[0]["movie_id"] == 11
    assert recommender.recommend(user_id=1, top_k=1, model_name="LightGCN")
    history = recommender.user_history(1, rating_store=store, top_k=5)
    assert [movie["movie_id"] for movie in history] == [12, 10]
    assert history[0]["user_rating"] == 4.5


def test_two_tower_artifact_mode(tmp_path) -> None:
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["Space One", "Space Two", "Kitchen Drama"],
            "genres": ["Sci-Fi", "Sci-Fi", "Drama"],
        }
    )
    save_artifact_bundle(
        tmp_path,
        catalog=catalog,
        user_mapping={1: 0},
        item_mapping={10: 0, 11: 1, 12: 2},
        content_embeddings=np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32),
        user_profiles=np.array([[0, 1]], dtype=np.float32),
        item_popularity=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        metrics={},
        hybrid_config={"two_tower_weight": 1.0, "content_weight": 0.0, "popularity_weight": 0.0},
        two_tower_user_embeddings=np.array([[1, 0]], dtype=np.float32),
        two_tower_item_embeddings=np.array([[1, 0], [0.8, 0.2], [0, 1]], dtype=np.float32),
    )

    recommender = HybridArtifactRecommender.from_dir(tmp_path)
    assert recommender.model_info()["has_two_tower"]
    results = recommender.recommend(user_id=1, top_k=1, model_name="two_tower", exclude_seen=False)
    assert results[0]["movie_id"] == 10
