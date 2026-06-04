"""LightGCN implementation for collaborative movie recommendation."""

from __future__ import annotations

import numpy as np


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    torch = None
    nn = None


def _require_torch():
    if torch is None:
        raise ImportError("LightGCN requires torch. Install requirements.txt first.")
    return torch


def build_normalized_adj(num_users: int, num_items: int, edges: np.ndarray, device: str = "cpu"):
    """Build the normalized bipartite adjacency matrix used by LightGCN."""
    torch_mod = _require_torch()
    if edges.size == 0:
        raise ValueError("LightGCN needs at least one training edge")

    user_nodes = torch_mod.as_tensor(edges[:, 0], dtype=torch_mod.long, device=device)
    item_nodes = torch_mod.as_tensor(edges[:, 1] + num_users, dtype=torch_mod.long, device=device)
    row = torch_mod.cat([user_nodes, item_nodes])
    col = torch_mod.cat([item_nodes, user_nodes])
    values = torch_mod.ones(row.shape[0], dtype=torch_mod.float32, device=device)

    num_nodes = num_users + num_items
    degree = torch_mod.zeros(num_nodes, dtype=torch_mod.float32, device=device)
    degree.index_add_(0, row, values)
    norm = torch_mod.pow(degree.clamp_min(1.0), -0.5)
    norm_values = norm[row] * values * norm[col]

    indices = torch_mod.stack([row, col])
    return torch_mod.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes), device=device).coalesce()


if torch is None:

    class LightGCNModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:

    class LightGCNModel(nn.Module):
        """Minimal LightGCN with ID embeddings and sparse propagation."""

        def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_dim: int = 64,
            num_layers: int = 3,
            adjacency=None,
        ) -> None:
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            self.num_layers = num_layers
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            self.adjacency = adjacency

        def propagate(self):
            if self.adjacency is None:
                raise ValueError("LightGCNModel.adjacency must be set before forward propagation")
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            embeddings_per_layer = [all_embeddings]
            current = all_embeddings
            for _ in range(self.num_layers):
                current = torch.sparse.mm(self.adjacency, current)
                embeddings_per_layer.append(current)
            final = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
            return torch.split(final, [self.num_users, self.num_items], dim=0)

        def forward(self, users, items):
            user_embeddings, item_embeddings = self.propagate()
            selected_users = user_embeddings[users]
            selected_items = item_embeddings[items]
            return (selected_users * selected_items).sum(dim=-1)

        def score_all_items(self, users):
            user_embeddings, item_embeddings = self.propagate()
            return user_embeddings[users] @ item_embeddings.T


def train_lightgcn_bpr(
    model,
    user_positive_items: dict[int, set[int]],
    num_items: int,
    epochs: int = 10,
    batch_size: int = 4096,
    samples_per_epoch: int | None = None,
    learning_rate: float = 1e-3,
    reg_weight: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
) -> list[float]:
    """Train LightGCN with BPR loss and return epoch losses."""
    torch_mod = _require_torch()
    from recommender.models.losses import bpr_loss, sample_bpr_triplets

    model.to(device)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    samples_per_epoch = samples_per_epoch or max(batch_size, sum(len(v) for v in user_positive_items.values()))
    losses: list[float] = []

    for _ in range(epochs):
        users, positives, negatives = sample_bpr_triplets(user_positive_items, num_items, samples_per_epoch, rng)
        epoch_losses: list[float] = []
        order = rng.permutation(len(users))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_users = torch_mod.as_tensor(users[idx], dtype=torch_mod.long, device=device)
            batch_pos = torch_mod.as_tensor(positives[idx], dtype=torch_mod.long, device=device)
            batch_neg = torch_mod.as_tensor(negatives[idx], dtype=torch_mod.long, device=device)

            optimizer.zero_grad()
            pos_scores = model(batch_users, batch_pos)
            neg_scores = model(batch_users, batch_neg)
            reg = (
                model.user_embedding(batch_users).norm(2).pow(2)
                + model.item_embedding(batch_pos).norm(2).pow(2)
                + model.item_embedding(batch_neg).norm(2).pow(2)
            ) / max(1, len(batch_users))
            loss = bpr_loss(pos_scores, neg_scores, reg_loss=reg, reg_weight=reg_weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    return losses
