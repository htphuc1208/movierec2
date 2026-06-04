"""Ranking and rating metrics for recommender evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_recall_ndcg_mrr_at_k(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []

    for user, truth in ground_truth.items():
        if not truth:
            continue
        top_items = recommendations.get(user, [])[:k]
        hits = np.array([1.0 if item in truth else 0.0 for item in top_items], dtype=np.float64)

        precisions.append(float(hits.sum() / max(k, 1)))
        recalls.append(float(hits.sum() / len(truth)))

        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = float((hits * discounts).sum())
        ideal_len = min(len(truth), k)
        ideal = float((1.0 / np.log2(np.arange(2, ideal_len + 2))).sum())
        ndcgs.append(dcg / ideal if ideal > 0 else 0.0)

        first_hit = np.where(hits > 0)[0]
        mrrs.append(float(1.0 / (first_hit[0] + 1)) if first_hit.size else 0.0)

    if not precisions:
        return {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, "mrr": 0.0}

    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "mrr": float(np.mean(mrrs)),
    }


def minmax(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mins = np.min(values, axis=axis, keepdims=True)
    maxs = np.max(values, axis=axis, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-8)
    return (values - mins) / denom


def top_k_from_scores(scores: np.ndarray, k: int) -> list[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    candidate_idx = np.argpartition(-scores, kth=k - 1)[:k]
    ordered = candidate_idx[np.argsort(-scores[candidate_idx])]
    return [int(item) for item in ordered]


def evaluate_score_fn(
    num_users: int,
    num_items: int,
    score_fn: Callable[[np.ndarray], np.ndarray],
    train_user_items: dict[int, set[int]],
    ground_truth_user_items: dict[int, set[int]],
    k: int = 10,
    batch_size: int = 512,
) -> dict[str, float]:
    """Evaluate a batched score function while masking training items."""
    recommendations: dict[int, list[int]] = {}
    users = sorted(ground_truth_user_items.keys())

    for start in range(0, len(users), batch_size):
        batch_users = np.array(users[start : start + batch_size], dtype=np.int64)
        scores = score_fn(batch_users)
        if scores.shape != (len(batch_users), num_items):
            raise ValueError(f"score_fn returned {scores.shape}, expected {(len(batch_users), num_items)}")

        scores = scores.copy()
        for row_idx, user in enumerate(batch_users):
            seen = train_user_items.get(int(user), set())
            if seen:
                scores[row_idx, list(seen)] = -np.inf
            recommendations[int(user)] = top_k_from_scores(scores[row_idx], k)

    return precision_recall_ndcg_mrr_at_k(recommendations, ground_truth_user_items, k=k)
