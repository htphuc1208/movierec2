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


def test_session_context_uses_two_tower_without_user(tmp_path) -> None:
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12],
            "title": ["Space One", "Space Followup", "Kitchen Drama"],
            "genres": ["Sci-Fi", "Sci-Fi", "Drama"],
            "tmdb_id": [100, 101, 102],
        }
    )
    save_artifact_bundle(
        tmp_path,
        catalog=catalog,
        user_mapping={},
        item_mapping={10: 0, 11: 1, 12: 2},
        content_embeddings=np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32),
        user_profiles=np.zeros((0, 2), dtype=np.float32),
        item_popularity=np.array([0.1, 0.1, 0.1], dtype=np.float32),
        metrics={},
        hybrid_config={"two_tower_weight": 1.0, "content_weight": 0.0, "popularity_weight": 0.0},
        two_tower_user_embeddings=np.zeros((0, 2), dtype=np.float32),
        two_tower_item_embeddings=np.array([[1, 0], [0.95, 0.05], [0, 1]], dtype=np.float32),
    )

    recommender = HybridArtifactRecommender.from_dir(tmp_path)
    results = recommender.recommend(top_k=1, session_context=["tmdb_100"], model_name="hybrid")
    assert results[0]["movie_id"] == 11


def test_duplicate_catalog_entries_are_hidden_from_lists(tmp_path) -> None:
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13, 14],
            "title": ["Alpha (2000)", "Alpha", "Beta", "Gamma (2001)", "Gamma"],
            "genres": ["Drama", "Drama", "Drama", "Comedy", "Comedy"],
            "tmdb_id": [100, 100, 101, np.nan, np.nan],
            "release_year": [2000, 2000, 2002, 2001, 2001],
            "vote_count": [500, 400, 300, 200, 100],
            "popularity": [5, 4, 3, 2, 1],
            "vote_average": [8.0, 7.9, 7.8, 7.7, 7.6],
        }
    )
    save_artifact_bundle(
        tmp_path,
        catalog=catalog,
        user_mapping={1: 0},
        item_mapping={10: 0, 11: 1, 12: 2, 13: 3, 14: 4},
        content_embeddings=np.array([[1, 0], [0.99, 0.01], [0.8, 0.2], [0, 1], [0.01, 0.99]], dtype=np.float32),
        user_profiles=np.array([[1, 0]], dtype=np.float32),
        item_popularity=np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32),
        metrics={},
        hybrid_config={"content_weight": 1.0, "popularity_weight": 0.0, "train_user_items": {"0": [0]}},
    )

    recommender = HybridArtifactRecommender.from_dir(tmp_path)

    assert [movie["movie_id"] for movie in recommender.search_movies("alpha", limit=5)] == [10]
    assert [movie["movie_id"] for movie in recommender.search_movies("gamma", limit=5)] == [13]

    recommendations = recommender.recommend(user_id=1, top_k=5, exclude_seen=True)
    assert all(movie["tmdb_id"] != 100 for movie in recommendations)
    assert len(_dedupe_keys(recommendations)) == len(set(_dedupe_keys(recommendations)))

    similar = recommender.similar_movies(10, top_k=5)
    assert all(movie["tmdb_id"] != 100 for movie in similar)
    assert len(_dedupe_keys(similar)) == len(set(_dedupe_keys(similar)))

    trending = recommender.trending_movies(top_k=5)
    assert [movie["movie_id"] for movie in trending] == [10, 12, 13]


def _dedupe_keys(movies: list[dict]) -> list[tuple[str, str]]:
    keys = []
    for movie in movies:
        if movie.get("tmdb_id") is not None:
            keys.append(("tmdb", str(movie["tmdb_id"])))
        else:
            title = str(movie["title"]).lower().replace(" (2001)", "").strip()
            keys.append(("title_year", f"{title}|{movie.get('release_year') or ''}"))
    return keys
