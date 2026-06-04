"""SVD baseline for comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recommender.data.movielens import build_sparse_interaction_matrix
from recommender.eval.metrics import rmse


def fit_svd_baseline(
    train: pd.DataFrame,
    num_users: int,
    num_items: int,
    n_components: int = 64,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    """Fit TruncatedSVD on an encoded user-item matrix."""
    matrix = build_sparse_interaction_matrix(train, num_users, num_items, value_col="rating")
    max_components = max(1, min(n_components, min(matrix.shape) - 1))
    model = TruncatedSVD(n_components=max_components, random_state=random_state)
    user_factors = model.fit_transform(matrix).astype(np.float32)
    item_factors = model.components_.T.astype(np.float32)
    return user_factors, item_factors, model


def predict_pairs(user_factors: np.ndarray, item_factors: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    if pairs.empty:
        return np.array([], dtype=np.float32)
    users = pairs["user_idx"].to_numpy(dtype=np.int64)
    items = pairs["item_idx"].to_numpy(dtype=np.int64)
    return np.sum(user_factors[users] * item_factors[items], axis=1)


def evaluate_svd_rmse(user_factors: np.ndarray, item_factors: np.ndarray, test: pd.DataFrame) -> float:
    if test.empty:
        return 0.0
    predictions = predict_pairs(user_factors, item_factors, test)
    return rmse(test["rating"].to_numpy(dtype=np.float32), predictions)
