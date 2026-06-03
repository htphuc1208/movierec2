from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse


DEFAULT_SEED = 42
DEFAULT_TOP_K = [10, 20]
DEFAULT_POSITIVE_THRESHOLD = 4.0
PROTOCOL_NAME = "per_user_temporal_80_10_10_pos4_full"


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_train_val_test_split(
    ratings: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> SplitFrames:
    """Split each user's interactions by timestamp with at least one val/test row when possible."""

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"])

    for _, user_rows in ordered.groupby("userId", sort=False):
        count = len(user_rows)
        if count < 3:
            train_parts.append(user_rows)
            continue
        test_size = max(1, int(round(count * test_ratio)))
        val_size = max(1, int(round(count * val_ratio)))
        train_end = max(1, count - val_size - test_size)
        val_end = count - test_size
        train_parts.append(user_rows.iloc[:train_end])
        val_parts.append(user_rows.iloc[train_end:val_end])
        test_parts.append(user_rows.iloc[val_end:])

    columns = ratings.columns
    return SplitFrames(
        train=_concat_or_empty(train_parts, columns),
        validation=_concat_or_empty(val_parts, columns),
        test=_concat_or_empty(test_parts, columns),
    )


def split_warm_cold_items(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_movie_ids = set(train["movieId"].astype(int).unique().tolist())
    warm_mask = holdout["movieId"].astype(int).isin(train_movie_ids)
    return holdout.loc[warm_mask].copy(), holdout.loc[~warm_mask].copy()


def relevant_by_user(frame: pd.DataFrame, positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD) -> dict[int, set[int]]:
    positives = frame.loc[pd.to_numeric(frame["rating"], errors="coerce") >= positive_threshold]
    grouped = positives.groupby("userId")["movieId"].apply(lambda values: set(int(value) for value in values))
    return {int(user_id): values for user_id, values in grouped.items()}


def evaluate_recommendations(
    recommendations_by_user: dict[int, Iterable[int]],
    holdout: pd.DataFrame,
    top_k: int,
    positive_threshold: float = DEFAULT_POSITIVE_THRESHOLD,
) -> dict[str, float]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []

    relevant_map = relevant_by_user(holdout, positive_threshold)
    all_holdout_users = sorted(set(int(uid) for uid in holdout["userId"].unique()))
    for user_id in all_holdout_users:
        relevant = relevant_map.get(user_id, set())
        if not relevant:
            precision_values.append(0.0)
            recall_values.append(0.0)
            ndcg_values.append(0.0)
            mrr_values.append(0.0)
            continue
        recommended = list(recommendations_by_user.get(int(user_id), []))
        precision_values.append(precision_at_k(recommended, relevant, top_k))
        recall_values.append(recall_at_k(recommended, relevant, top_k))
        ndcg_values.append(ndcg_at_k(recommended, relevant, top_k))
        mrr_values.append(mrr_at_k(recommended, relevant, top_k))

    return {
        f"precision@{top_k}": _mean(precision_values),
        f"recall@{top_k}": _mean(recall_values),
        f"ndcg@{top_k}": _mean(ndcg_values),
        f"mrr@{top_k}": _mean(mrr_values),
    }


def rating_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    true = [float(value) for value in y_true]
    pred = [float(value) for value in y_pred]
    if not true:
        return {"rmse": 0.0, "mae": 0.0}
    absolute_error = [abs(a - b) for a, b in zip(true, pred)]
    return {"rmse": rmse(true, pred), "mae": _mean(absolute_error)}


def _concat_or_empty(parts: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(parts, ignore_index=True)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
