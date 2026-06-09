"""Matrix-factorization style models for comparison experiments."""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recommender.models.base import ModelSkip


@dataclass
class BPRMFRecommender:
    embedding_dim: int = 64
    epochs: int = 10
    batch_size: int = 4096
    learning_rate: float = 1e-3
    reg_weight: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    name: str = "bpr_mf"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "BPRMFRecommender":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ModelSkip("torch is not installed") from exc

        from recommender.models.losses import bpr_loss, sample_bpr_triplets

        class _BPRMF(nn.Module):
            def __init__(self, num_users: int, num_items: int, dim: int) -> None:
                super().__init__()
                self.user_embedding = nn.Embedding(num_users, dim)
                self.item_embedding = nn.Embedding(num_items, dim)
                nn.init.normal_(self.user_embedding.weight, std=0.1)
                nn.init.normal_(self.item_embedding.weight, std=0.1)

            def forward(self, users, items):
                return (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)

        model = _BPRMF(dataset.num_users, dataset.num_items, self.embedding_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        samples_per_epoch = max(self.batch_size, sum(len(v) for v in dataset.train_user_items.values()))
        losses: list[float] = []
        verbose = os.getenv("RECOMMENDER_VERBOSE_TRAIN", "0") == "1"
        for epoch in range(self.epochs):
            users, positives, negatives = sample_bpr_triplets(dataset.train_user_items, dataset.num_items, samples_per_epoch, rng)
            order = rng.permutation(len(users))
            batch_losses: list[float] = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_users = torch.as_tensor(users[idx], dtype=torch.long, device=self.device)
                batch_pos = torch.as_tensor(positives[idx], dtype=torch.long, device=self.device)
                batch_neg = torch.as_tensor(negatives[idx], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                pos_scores = model(batch_users, batch_pos)
                neg_scores = model(batch_users, batch_neg)
                reg = (
                    model.user_embedding(batch_users).norm(2).pow(2)
                    + model.item_embedding(batch_pos).norm(2).pow(2)
                    + model.item_embedding(batch_neg).norm(2).pow(2)
                ) / max(1, len(batch_users))
                loss = bpr_loss(pos_scores, neg_scores, reg_loss=reg, reg_weight=self.reg_weight)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if verbose:
                print(f"{self.name} epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.6f}", flush=True)
        self.user_embeddings_ = model.user_embedding.weight.detach().cpu().numpy().astype(np.float32)
        self.item_embeddings_ = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
        self.metadata = {"embedding_dim": self.embedding_dim, "epochs": self.epochs, "losses": losses}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_embeddings_[user_indices] @ self.item_embeddings_.T


@dataclass
class LightGCNRecommender:
    embedding_dim: int = 64
    num_layers: int = 3
    epochs: int = 10
    batch_size: int = 4096
    device: str = "cpu"
    seed: int = 42
    name: str = "lightgcn_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "LightGCNRecommender":
        try:
            import torch
        except ImportError as exc:
            raise ModelSkip("torch is not installed") from exc
        from recommender.models.lightgcn import LightGCNModel, build_normalized_adj, train_lightgcn_bpr

        edges = dataset.train[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        adjacency = build_normalized_adj(dataset.num_users, dataset.num_items, edges, device=self.device)
        model = LightGCNModel(
            dataset.num_users,
            dataset.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            adjacency=adjacency,
        )
        losses = train_lightgcn_bpr(
            model,
            dataset.train_user_items,
            dataset.num_items,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
            device=self.device,
        )
        model.eval()
        with torch.no_grad():
            users, items = model.propagate()
        self.user_embeddings_ = users.detach().cpu().numpy().astype(np.float32)
        self.item_embeddings_ = items.detach().cpu().numpy().astype(np.float32)
        self.metadata = {"embedding_dim": self.embedding_dim, "layers": self.num_layers, "epochs": self.epochs, "losses": losses}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_embeddings_[user_indices] @ self.item_embeddings_.T


@dataclass
class ImplicitALSRecommender:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 20
    name: str = "implicit_als"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "ImplicitALSRecommender":
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError as exc:
            raise ModelSkip("implicit is not installed") from exc
        model = AlternatingLeastSquares(factors=self.factors, regularization=self.regularization, iterations=self.iterations)
        model.fit(dataset.train_matrix.T.tocsr())
        self.user_factors_ = model.user_factors.astype(np.float32)
        self.item_factors_ = model.item_factors.astype(np.float32)
        self.metadata = {"factors": self.factors, "iterations": self.iterations}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors_[user_indices] @ self.item_factors_.T


@dataclass
class LightFMWARPRecommender:
    no_components: int = 64
    epochs: int = 10
    random_state: int = 42
    name: str = "lightfm_warp"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "LightFMWARPRecommender":
        try:
            from lightfm import LightFM
        except ImportError as exc:
            raise ModelSkip("lightfm is not installed") from exc
        model = LightFM(no_components=self.no_components, loss="warp", random_state=self.random_state)
        model.fit(dataset.train_matrix, epochs=self.epochs, num_threads=1)
        self.model_ = model
        self.num_items_ = dataset.num_items
        self.metadata = {"no_components": self.no_components, "epochs": self.epochs}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        item_ids = np.arange(self.num_items_, dtype=np.int32)
        scores = []
        for user in user_indices:
            scores.append(self.model_.predict(int(user), item_ids).astype(np.float32))
        return np.vstack(scores)


@dataclass
class NeuMFRecommender:
    embedding_dim: int = 32
    hidden_dim: int = 64
    epochs: int = 5
    batch_size: int = 4096
    learning_rate: float = 1e-3
    negatives_per_positive: int = 2
    seed: int = 42
    device: str = "cpu"
    name: str = "neumf"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "NeuMFRecommender":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ModelSkip("torch is not installed") from exc

        class _NeuMF(nn.Module):
            def __init__(self, num_users: int, num_items: int, dim: int, hidden: int) -> None:
                super().__init__()
                self.gmf_user = nn.Embedding(num_users, dim)
                self.gmf_item = nn.Embedding(num_items, dim)
                self.mlp_user = nn.Embedding(num_users, dim)
                self.mlp_item = nn.Embedding(num_items, dim)
                self.mlp = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
                self.out = nn.Linear(dim + hidden // 2, 1)

            def forward(self, users, items):
                gmf = self.gmf_user(users) * self.gmf_item(items)
                mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
                return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

        model = _NeuMF(dataset.num_users, dataset.num_items, self.embedding_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        users, items, labels = _sample_binary_examples(dataset, self.negatives_per_positive, self.seed)
        rng = np.random.default_rng(self.seed)
        losses: list[float] = []
        verbose = os.getenv("RECOMMENDER_VERBOSE_TRAIN", "0") == "1"
        for epoch in range(self.epochs):
            order = rng.permutation(len(users))
            batch_losses: list[float] = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_users = torch.as_tensor(users[idx], dtype=torch.long, device=self.device)
                batch_items = torch.as_tensor(items[idx], dtype=torch.long, device=self.device)
                batch_labels = torch.as_tensor(labels[idx], dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                loss = criterion(model(batch_users, batch_items), batch_labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            epoch_loss = float(np.mean(batch_losses))
            losses.append(epoch_loss)
            if verbose:
                print(f"{self.name} epoch {epoch + 1}/{self.epochs} loss={epoch_loss:.6f}", flush=True)
        self.model_ = model
        self.num_items_ = dataset.num_items
        self.metadata = {"embedding_dim": self.embedding_dim, "epochs": self.epochs, "losses": losses}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        item_ids = torch.arange(self.num_items_, dtype=torch.long, device=self.device)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for user in user_indices:
                users = torch.full((self.num_items_,), int(user), dtype=torch.long, device=self.device)
                scores.append(self.model_(users, item_ids).detach().cpu().numpy().astype(np.float32))
        return np.vstack(scores)


def _sample_binary_examples(dataset, negatives_per_positive: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive_pairs = dataset.train[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
    users = [positive_pairs[:, 0]]
    items = [positive_pairs[:, 1]]
    labels = [np.ones(len(positive_pairs), dtype=np.float32)]
    neg_users: list[int] = []
    neg_items: list[int] = []
    for user, _ in positive_pairs:
        seen = dataset.train_user_items[int(user)]
        for _ in range(negatives_per_positive):
            item = int(rng.integers(0, dataset.num_items))
            while item in seen:
                item = int(rng.integers(0, dataset.num_items))
            neg_users.append(int(user))
            neg_items.append(item)
    if neg_users:
        users.append(np.asarray(neg_users, dtype=np.int64))
        items.append(np.asarray(neg_items, dtype=np.int64))
        labels.append(np.zeros(len(neg_users), dtype=np.float32))
    return np.concatenate(users), np.concatenate(items), np.concatenate(labels)
