"""Hybrid weighted and learned rankers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recommender.eval.metrics import evaluate_score_fn, minmax


@dataclass
class WeightedHybridRecommender:
    components: list[Any]
    include_popularity: bool = True
    tune: bool = True
    k: int = 10
    name: str = "hybrid_weighted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "WeightedHybridRecommender":
        self.dataset_ = dataset
        self.popularity_ = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        if self.popularity_.max() > 0:
            self.popularity_ = self.popularity_ / self.popularity_.max()
        component_count = len(self.components) + (1 if self.include_popularity else 0)
        if not self.tune or not dataset.val_user_items:
            self.weights_ = np.ones(component_count, dtype=np.float32) / max(1, component_count)
            self.metadata = {"weights": self.weights_.tolist(), "tuned": False}
            return self
        candidates = _weight_grid(component_count)
        best_weights = candidates[0]
        best_ndcg = -1.0
        for weights in candidates:
            metrics = evaluate_score_fn(
                dataset.num_users,
                dataset.num_items,
                lambda users, w=weights: self._score_with_weights(users, w),
                dataset.train_user_items,
                dataset.val_user_items,
                k=self.k,
            )
            ndcg = metrics.get(f"ndcg@{self.k}", 0.0)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_weights = weights
        self.weights_ = best_weights.astype(np.float32)
        self.metadata = {"weights": self.weights_.tolist(), "tuned": True, f"validation_ndcg@{self.k}": best_ndcg}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self._score_with_weights(user_indices, self.weights_)

    def _score_with_weights(self, user_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(user_indices), self.dataset_.num_items), dtype=np.float32)
        offset = 0
        for component in self.components:
            scores += float(weights[offset]) * minmax(component.score_users(user_indices), axis=1)
            offset += 1
        if self.include_popularity:
            scores += float(weights[offset]) * minmax(np.broadcast_to(self.popularity_[None, :], scores.shape), axis=1)
        return scores


@dataclass
class SGDRankHybridRecommender:
    components: list[Any]
    include_popularity: bool = True
    negatives_per_positive: int = 2
    max_train_samples: int = 200_000
    seed: int = 42
    name: str = "hybrid_ranker"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "SGDRankHybridRecommender":
        self.dataset_ = dataset
        self.popularity_ = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        if self.popularity_.max() > 0:
            self.popularity_ = self.popularity_ / self.popularity_.max()
        users, items, labels = self._sample_pairs(dataset)
        features = self._features_for_pairs(users, items)
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=self.seed),
        )
        self.pipeline_.fit(features, labels)
        self.metadata = {
            "components": [component.name for component in self.components],
            "include_popularity": self.include_popularity,
            "train_samples": int(len(labels)),
        }
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        all_items = np.arange(self.dataset_.num_items, dtype=np.int64)
        for user in user_indices:
            users = np.full(self.dataset_.num_items, int(user), dtype=np.int64)
            features = self._features_for_pairs(users, all_items)
            rows.append(self.pipeline_.predict_proba(features)[:, 1].astype(np.float32))
        return np.vstack(rows)

    def _sample_pairs(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positives = dataset.train[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        if len(positives) > self.max_train_samples:
            positives = positives[rng.choice(len(positives), size=self.max_train_samples, replace=False)]
        neg_users: list[int] = []
        neg_items: list[int] = []
        for user, _ in positives:
            seen = dataset.train_user_items[int(user)]
            for _ in range(self.negatives_per_positive):
                item = int(rng.integers(0, dataset.num_items))
                while item in seen:
                    item = int(rng.integers(0, dataset.num_items))
                neg_users.append(int(user))
                neg_items.append(item)
        users = np.concatenate([positives[:, 0], np.asarray(neg_users, dtype=np.int64)])
        items = np.concatenate([positives[:, 1], np.asarray(neg_items, dtype=np.int64)])
        labels = np.concatenate([np.ones(len(positives), dtype=np.int64), np.zeros(len(neg_users), dtype=np.int64)])
        return users, items, labels

    def _features_for_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        unique_users, inverse = np.unique(users, return_inverse=True)
        for component in self.components:
            score_matrix = component.score_users(unique_users)
            columns.append(score_matrix[inverse, items])
        if self.include_popularity:
            columns.append(self.popularity_[items])
        columns.append(np.asarray([len(self.dataset_.train_user_items.get(int(user), set())) for user in users], dtype=np.float32))
        return np.vstack(columns).T.astype(np.float32)


def _weight_grid(component_count: int) -> list[np.ndarray]:
    if component_count <= 1:
        return [np.ones(1, dtype=np.float32)]
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    candidates: list[np.ndarray] = []
    if component_count == 2:
        for first in values:
            candidates.append(np.asarray([first, 1.0 - first], dtype=np.float32))
    elif component_count == 3:
        for first in values:
            for second in values:
                third = 1.0 - first - second
                if third >= 0:
                    candidates.append(np.asarray([first, second, third], dtype=np.float32))
    else:
        candidates.append(np.ones(component_count, dtype=np.float32) / component_count)
    return candidates
