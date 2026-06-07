"""Learned two-tower recommender with user embeddings and item metadata MLP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recommender.models.base import ModelSkip


@dataclass
class LearnedTwoTowerRecommender:
    embedding_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 10
    batch_size: int = 4096
    learning_rate: float = 1e-3
    reg_weight: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    embedding_attr: str = "content_embeddings"
    name: str = "learned_two_tower"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "LearnedTwoTowerRecommender":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ModelSkip("torch is not installed") from exc
        from recommender.models.losses import bpr_loss, sample_bpr_triplets

        item_features = getattr(dataset, self.embedding_attr)
        if item_features is None:
            raise ModelSkip(f"{self.embedding_attr} is unavailable")

        class _TwoTower(nn.Module):
            def __init__(self, num_users: int, feature_dim: int, dim: int, hidden: int) -> None:
                super().__init__()
                self.user_tower = nn.Embedding(num_users, dim)
                self.item_tower = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
                nn.init.normal_(self.user_tower.weight, std=0.1)

            def encode_items(self, features):
                return torch.nn.functional.normalize(self.item_tower(features), dim=-1)

            def encode_users(self, users):
                return torch.nn.functional.normalize(self.user_tower(users), dim=-1)

            def forward(self, users, item_features_batch):
                return (self.encode_users(users) * self.encode_items(item_features_batch)).sum(dim=-1)

        torch_features = torch.as_tensor(item_features, dtype=torch.float32, device=self.device)
        model = _TwoTower(dataset.num_users, item_features.shape[1], self.embedding_dim, self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        samples_per_epoch = max(self.batch_size, sum(len(v) for v in dataset.train_user_items.values()))
        losses: list[float] = []
        for _ in range(self.epochs):
            users, positives, negatives = sample_bpr_triplets(dataset.train_user_items, dataset.num_items, samples_per_epoch, rng)
            order = rng.permutation(len(users))
            batch_losses: list[float] = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch_users = torch.as_tensor(users[idx], dtype=torch.long, device=self.device)
                batch_pos = torch.as_tensor(positives[idx], dtype=torch.long, device=self.device)
                batch_neg = torch.as_tensor(negatives[idx], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                pos_scores = model(batch_users, torch_features[batch_pos])
                neg_scores = model(batch_users, torch_features[batch_neg])
                reg = model.user_tower(batch_users).norm(2).pow(2) / max(1, len(batch_users))
                loss = bpr_loss(pos_scores, neg_scores, reg_loss=reg, reg_weight=self.reg_weight)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            all_items = model.encode_items(torch_features).detach().cpu().numpy().astype(np.float32)
            all_users = model.encode_users(torch.arange(dataset.num_users, device=self.device)).detach().cpu().numpy().astype(np.float32)
        self.user_embeddings_ = all_users
        self.item_embeddings_ = all_items
        self.metadata = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "embedding_attr": self.embedding_attr,
            "losses": losses,
        }
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_embeddings_[user_indices] @ self.item_embeddings_.T
