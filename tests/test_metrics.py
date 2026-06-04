from __future__ import annotations

import numpy as np

from recommender.eval.metrics import evaluate_score_fn, precision_recall_ndcg_mrr_at_k, rmse, top_k_from_scores


def test_ranking_metrics() -> None:
    recs = {0: [1, 2, 3], 1: [4, 5, 6]}
    truth = {0: {2, 9}, 1: {4}}
    metrics = precision_recall_ndcg_mrr_at_k(recs, truth, k=3)

    assert 0 < metrics["precision@3"] < 1
    assert 0 < metrics["recall@3"] <= 1
    assert 0 < metrics["ndcg@3"] <= 1
    assert 0 < metrics["mrr"] <= 1


def test_rmse_and_top_k() -> None:
    assert rmse(np.array([1, 2, 3]), np.array([1, 2, 5])) > 0
    assert top_k_from_scores(np.array([0.1, 0.9, 0.2]), 2) == [1, 2]


def test_evaluate_score_fn_masks_train_items() -> None:
    scores = np.array([[0.9, 0.8, 0.1], [0.2, 0.3, 0.7]], dtype=np.float32)

    def score_fn(users: np.ndarray) -> np.ndarray:
        return scores[users]

    metrics = evaluate_score_fn(
        num_users=2,
        num_items=3,
        score_fn=score_fn,
        train_user_items={0: {0}},
        ground_truth_user_items={0: {1}, 1: {2}},
        k=1,
        batch_size=1,
    )
    assert metrics["precision@1"] == 1.0
