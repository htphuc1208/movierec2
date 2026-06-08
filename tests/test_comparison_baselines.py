from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from recommender.experiments.comparison import ComparisonConfig, ExperimentDataset, encode_item_texts_cached, fit_and_evaluate_model
from recommender.models.baselines import ContentAverageRecommender, EASERecommender, ItemKNNRecommender, PopularityRecommender, UserKNNRecommender
from recommender.models.matrix_factorization import _sample_binary_examples
from recommender.models.rankers import SGDRankHybridRecommender, StrongHybridRankerRecommender, WeightedHybridRecommender


def _dataset() -> ExperimentDataset:
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [0, 1, 1, 2, 2, 3],
            "userId": [10, 10, 11, 11, 12, 12],
            "movieId": [20, 21, 21, 22, 22, 23],
            "rating": [5.0] * 6,
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )
    val = pd.DataFrame({"user_idx": [0, 1, 2], "item_idx": [2, 3, 0]})
    test = pd.DataFrame({"user_idx": [0, 1, 2], "item_idx": [3, 0, 1]})
    matrix = sparse.csr_matrix((np.ones(len(train), dtype=np.float32), (train["user_idx"], train["item_idx"])), shape=(3, 4))
    catalog = pd.DataFrame(
        {
            "movieId": [20, 21, 22, 23],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Drama", "Action", "Action"],
        }
    )
    return ExperimentDataset(
        name="synthetic",
        train=train,
        val=val,
        test=test,
        catalog=catalog,
        train_matrix=matrix,
        train_user_items={0: {0, 1}, 1: {1, 2}, 2: {2, 3}},
        val_user_items={0: {2}, 1: {3}, 2: {0}},
        test_user_items={0: {3}, 1: {0}, 2: {1}},
        user_mapping={10: 0, 11: 1, 12: 2},
        item_mapping={20: 0, 21: 1, 22: 2, 23: 3},
        content_embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        content_embeddings_no_tmdb=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_classical_baselines_score_shape() -> None:
    dataset = _dataset()
    models = [
        PopularityRecommender(name="popularity_only"),
        ItemKNNRecommender(top_k=2),
        UserKNNRecommender(top_k=2),
        EASERecommender(max_items=10),
    ]
    users = np.asarray([0, 2], dtype=np.int64)
    for model in models:
        model.fit(dataset)
        scores = model.score_users(users)
        assert scores.shape == (2, dataset.num_items)
        assert np.isfinite(scores).all()


def test_bpr_binary_sampler_creates_negatives() -> None:
    users, items, labels = _sample_binary_examples(_dataset(), negatives_per_positive=2, seed=7)
    assert len(users) == len(items) == len(labels)
    assert labels.sum() == 6
    assert len(labels) == 18


def test_hybrid_rankers_evaluate_on_synthetic_dataset() -> None:
    dataset = _dataset()
    popularity = PopularityRecommender(name="popularity_only").fit(dataset)
    content = ContentAverageRecommender(name="tfidf_only").fit(dataset)
    config = ComparisonConfig(k=2, batch_size=2, epochs=1, max_ranker_samples=100)

    weighted = WeightedHybridRecommender([popularity, content], include_popularity=False, tune=True, k=2)
    _, weighted_row = fit_and_evaluate_model(dataset, weighted, config)
    assert weighted_row["status"] == "ok"
    assert "ndcg@2" in weighted_row["metrics"]

    ranker = SGDRankHybridRecommender([popularity, content], include_popularity=True, max_train_samples=100)
    _, ranker_row = fit_and_evaluate_model(dataset, ranker, config)
    assert ranker_row["status"] == "ok"
    assert "precision@2" in ranker_row["metrics"]

    strong = StrongHybridRankerRecommender([popularity, content], include_popularity=True, max_train_samples=100, ranker="sgd")
    _, strong_row = fit_and_evaluate_model(dataset, strong, config)
    assert strong_row["status"] == "ok"
    assert strong_row["metadata"]["ranker"] == "sgd_fallback"
    assert "genre_overlap" in strong_row["metadata"]["feature_names"]


def test_content_embedding_cache_roundtrip(tmp_path) -> None:
    dataset = _dataset()
    config = ComparisonConfig(content_backend="tfidf", embedding_cache_dir=tmp_path, use_content_cache=True)

    first = encode_item_texts_cached(dataset.catalog, dataset_name="synthetic", variant="full", config=config)
    second = encode_item_texts_cached(dataset.catalog, dataset_name="synthetic", variant="full", config=config)

    assert np.allclose(first, second)
    assert list(tmp_path.glob("synthetic_full_tfidf_*.npy"))
