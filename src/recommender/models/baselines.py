"""Classical recommender baselines for comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize

from recommender.models.base import ModelSkip


@dataclass
class RandomRecommender:
    seed: int = 42
    name: str = "random"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "RandomRecommender":
        rng = np.random.default_rng(self.seed)
        self.item_scores_ = rng.random(dataset.num_items).astype(np.float32)
        self.metadata = {"seed": self.seed}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_scores_[None, :], (len(user_indices), len(self.item_scores_))).copy()


@dataclass
class PopularityRecommender:
    name: str = "popularity"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "PopularityRecommender":
        counts = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        self.item_scores_ = np.log1p(counts)
        max_score = float(self.item_scores_.max()) if self.item_scores_.size else 0.0
        if max_score > 0:
            self.item_scores_ = self.item_scores_ / max_score
        self.metadata = {"nonzero_items": int(np.count_nonzero(counts))}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.item_scores_[None, :], (len(user_indices), len(self.item_scores_))).copy()


@dataclass
class ItemKNNRecommender:
    top_k: int = 100
    name: str = "item_knn_cosine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "ItemKNNRecommender":
        matrix = dataset.train_matrix.astype(np.float32)
        item_user = normalize(matrix.T, norm="l2", axis=1)
        similarity = (item_user @ item_user.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity_ = _topk_sparse_rows(similarity, self.top_k)
        self.train_matrix_ = matrix.tocsr()
        self.metadata = {"top_k": self.top_k, "similarity_nnz": int(self.similarity_.nnz)}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.train_matrix_[user_indices] @ self.similarity_).toarray().astype(np.float32)


@dataclass
class UserKNNRecommender:
    top_k: int = 100
    name: str = "user_knn_cosine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "UserKNNRecommender":
        matrix = dataset.train_matrix.astype(np.float32).tocsr()
        user_item = normalize(matrix, norm="l2", axis=1)
        similarity = (user_item @ user_item.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity_ = _topk_sparse_rows(similarity, self.top_k)
        self.train_matrix_ = matrix
        self.metadata = {"top_k": self.top_k, "similarity_nnz": int(self.similarity_.nnz)}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.similarity_[user_indices] @ self.train_matrix_).toarray().astype(np.float32)


@dataclass
class SVDRankingRecommender:
    n_components: int = 64
    random_state: int = 42
    name: str = "svd_ranking"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "SVDRankingRecommender":
        matrix = dataset.train_matrix.astype(np.float32)
        max_components = max(1, min(self.n_components, min(matrix.shape) - 1))
        if max_components < 1:
            raise ModelSkip("not enough users/items for SVD")
        model = TruncatedSVD(n_components=max_components, random_state=self.random_state)
        self.user_factors_ = model.fit_transform(matrix).astype(np.float32)
        self.item_factors_ = model.components_.T.astype(np.float32)
        self.metadata = {"n_components": max_components, "explained_variance": float(model.explained_variance_ratio_.sum())}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors_[user_indices] @ self.item_factors_.T


@dataclass
class EASERecommender:
    l2: float = 250.0
    max_items: int = 5000
    name: str = "ease"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "EASERecommender":
        if dataset.num_items > self.max_items:
            raise ModelSkip(f"EASE skipped because num_items={dataset.num_items} > max_items={self.max_items}")
        x = dataset.train_matrix.astype(np.float32)
        gram = (x.T @ x).toarray().astype(np.float64)
        diag = np.diag_indices_from(gram)
        gram[diag] += self.l2
        precision = np.linalg.inv(gram)
        weights = -precision / np.diag(precision)
        weights[diag] = 0.0
        self.weights_ = weights.astype(np.float32)
        self.train_matrix_ = x.tocsr()
        self.metadata = {"l2": self.l2}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.train_matrix_[user_indices] @ self.weights_).astype(np.float32)


@dataclass
class SLIMElasticNetRecommender:
    alpha: float = 1e-3
    l1_ratio: float = 0.1
    max_items: int = 1000
    max_iter: int = 500
    name: str = "slim_elasticnet"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "SLIMElasticNetRecommender":
        if dataset.num_items > self.max_items:
            raise ModelSkip(f"SLIM skipped because num_items={dataset.num_items} > max_items={self.max_items}")
        x = dataset.train_matrix.astype(np.float32).tocsc()
        weights = np.zeros((dataset.num_items, dataset.num_items), dtype=np.float32)
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            max_iter=self.max_iter,
            selection="random",
            random_state=42,
        )
        for item in range(dataset.num_items):
            y = x[:, item].toarray().ravel()
            if np.count_nonzero(y) == 0:
                continue
            x_work = x.copy()
            start, end = x_work.indptr[item], x_work.indptr[item + 1]
            x_work.data[start:end] = 0.0
            model.fit(x_work, y)
            weights[:, item] = model.sparse_coef_.toarray().ravel().astype(np.float32)
        self.weights_ = weights
        self.train_matrix_ = dataset.train_matrix.astype(np.float32).tocsr()
        self.metadata = {"alpha": self.alpha, "l1_ratio": self.l1_ratio, "max_items": self.max_items}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.train_matrix_[user_indices] @ self.weights_).astype(np.float32)


@dataclass
class ContentAverageRecommender:
    name: str = "content_average"
    embedding_attr: str = "content_embeddings"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "ContentAverageRecommender":
        item_embeddings = getattr(dataset, self.embedding_attr)
        if item_embeddings is None:
            raise ModelSkip(f"{self.embedding_attr} is unavailable")
        self.item_embeddings_ = item_embeddings.astype(np.float32)
        dim = self.item_embeddings_.shape[1]
        profiles = np.zeros((dataset.num_users, dim), dtype=np.float32)
        counts = np.zeros(dataset.num_users, dtype=np.float32)
        for row in dataset.train[["user_idx", "item_idx"]].itertuples(index=False):
            profiles[int(row.user_idx)] += self.item_embeddings_[int(row.item_idx)]
            counts[int(row.user_idx)] += 1.0
        nonzero = counts > 0
        profiles[nonzero] /= counts[nonzero, None]
        self.user_profiles_ = normalize(profiles).astype(np.float32)
        self.metadata = {"embedding_attr": self.embedding_attr}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_profiles_[user_indices] @ self.item_embeddings_.T


def _topk_sparse_rows(matrix: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    if top_k <= 0:
        return matrix.tocsr()
    matrix = matrix.tocsr()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for row_idx in range(matrix.shape[0]):
        start, end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_cols = matrix.indices[start:end]
        row_vals = matrix.data[start:end]
        if row_vals.size > top_k:
            keep = np.argpartition(-row_vals, top_k - 1)[:top_k]
            row_cols = row_cols[keep]
            row_vals = row_vals[keep]
        rows.append(np.full(row_vals.shape, row_idx, dtype=np.int32))
        cols.append(row_cols.astype(np.int32))
        vals.append(row_vals.astype(np.float32))
    if not vals:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=matrix.shape)
