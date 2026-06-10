"""LightGCN implementation for collaborative movie recommendation."""

from __future__ import annotations

import os
import time

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

# tao ma tran ke chuan hoa A~ = D^-0.5 @ A @ D^-0.5, trong do D la ma tran bac cua A
def build_normalized_adj(num_users: int, num_items: int, edges: np.ndarray, device: str = "cpu"):
    """Build the normalized bipartite adjacency matrix used by LightGCN."""
    
    # edges la ma tran 2 cot [user_idx, item_idx] voi item_idx da duoc 
    # dich chuyen de khong trung voi user_idx
    torch_mod = _require_torch()
    if edges.size == 0:
        raise ValueError("LightGCN needs at least one training edge")

    user_nodes = torch_mod.as_tensor(edges[:, 0], dtype=torch_mod.long, device=device)
    item_nodes = torch_mod.as_tensor(edges[:, 1] + num_users, dtype=torch_mod.long, device=device)
    # row dai dien cho tat ca cac nut nguon, cat = concat
    # col dai dien cho tat ca cac nut dich, cat = concat
    row = torch_mod.cat([user_nodes, item_nodes])
    col = torch_mod.cat([item_nodes, user_nodes])
    # values la vector 1 voi do dai bang so luong canh, gia tri 1 dai dien cho moi canh
    values = torch_mod.ones(row.shape[0], dtype=torch_mod.float32, device=device)

    num_nodes = num_users + num_items
    # tinh bac cua tung nut: D[k] = sum_{i} A[i,k], voi A[i,k] = 1 neu co canh i-k, 0 neu khong
    degree = torch_mod.zeros(num_nodes, dtype=torch_mod.float32, device=device)
    degree.index_add_(0, row, values)
    # tinh he so chuan hoa cho tung canh: norm[i] * norm[j] voi i,j la 2 dau canh, norm[k] = D[k]^-0.5
    norm = torch_mod.pow(degree.clamp_min(1.0), -0.5)
    norm_values = norm[row] * values * norm[col]
    # tao ma tran ke chuan hoa dang COO sparse tensor voi indices = [row, col], values = norm_values, shape = (num_nodes, num_nodes)
    indices = torch_mod.stack([row, col])
    return torch_mod.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes), device=device).coalesce()


if torch is None:

    class LightGCNModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()

else:
    # Minimal LightGCN implementation with user/item ID embeddings and sparse propagation.
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
            # concatenate user and item embeddings to shape (num_users + num_items, embedding_dim)
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            # embeddings_per_layer la danh sach cac embedding tai moi lop
            embeddings_per_layer = [all_embeddings]
            current = all_embeddings
            for _ in range(self.num_layers):
                # propagate embeddings qua ma tran ke chuan hoa: current = A~ @ current, 
                # voi A~ la ma tran ke chuan hoa da duoc tinh truoc
                current = torch.sparse.mm(self.adjacency, current)
                embeddings_per_layer.append(current)
            # trung binh cac embedding tai cac lop de duoc embedding cuoi cung, 
            # sau do tach lai thanh user va item embeddings
            final = torch.stack(embeddings_per_layer, dim=0).mean(dim=0)
            return torch.split(final, [self.num_users, self.num_items], dim=0)

        def forward(self, users, items):
            user_embeddings, item_embeddings = self.propagate()
            selected_users = user_embeddings[users]
            selected_items = item_embeddings[items]
            # score la tich vo huong giua embedding nguoi dung va san pham: 
            # score[u,i] = <embedding_user[u], embedding_item[i]>
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

    verbose = os.getenv("RECOMMENDER_VERBOSE_TRAIN", "0") == "1"
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        # sample_bpr_triplets tra ve 3 vector: users, positives, negatives, moi vector co do dai bang samples_per_epoch
        users, positives, negatives = sample_bpr_triplets(user_positive_items, num_items, samples_per_epoch, rng)
        epoch_losses: list[float] = []
        order = rng.permutation(len(users))
        batch_count = int(np.ceil(len(order) / max(1, batch_size)))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch_users = torch_mod.as_tensor(users[idx], dtype=torch_mod.long, device=device)
            batch_pos = torch_mod.as_tensor(positives[idx], dtype=torch_mod.long, device=device)
            batch_neg = torch_mod.as_tensor(negatives[idx], dtype=torch_mod.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            user_embeddings, item_embeddings = model.propagate()
            selected_users = user_embeddings[batch_users]
            pos_items = item_embeddings[batch_pos]
            neg_items = item_embeddings[batch_neg]
            # score la tich vo huong giua embedding nguoi dung va san pham: 
            # score[u,i] = <embedding_user[u], embedding_item[i]>
            pos_scores = (selected_users * pos_items).sum(dim=-1)
            neg_scores = (selected_users * neg_items).sum(dim=-1)
            # l2 regularization tren embedding nguoi dung va san pham 
            # trong batch, chia cho kich thuoc batch de co do on dinh hon
            reg = (
                model.user_embedding(batch_users).norm(2).pow(2)
                + model.item_embedding(batch_pos).norm(2).pow(2)
                + model.item_embedding(batch_neg).norm(2).pow(2)
            ) / max(1, len(batch_users))
            loss = bpr_loss(pos_scores, neg_scores, reg_loss=reg, reg_weight=reg_weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)
        if verbose:
            if str(device).startswith("cuda"):
                torch_mod.cuda.synchronize()
                cuda_mem = torch_mod.cuda.max_memory_allocated() / (1024**2)
                print(
                    f"LightGCN epoch {epoch + 1}/{epochs} loss={epoch_loss:.6f} "
                    f"seconds={time.perf_counter() - epoch_start:.2f} batches={batch_count} "
                    f"cuda_max_mem_mib={cuda_mem:.0f}",
                    flush=True,
                )
            else:
                print(
                    f"LightGCN epoch {epoch + 1}/{epochs} loss={epoch_loss:.6f} "
                    f"seconds={time.perf_counter() - epoch_start:.2f} batches={batch_count}",
                    flush=True,
                )

    return losses
