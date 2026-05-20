from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if torch is not None:

    class LightGCNModel(nn.Module):
        """Minimal LightGCN for bipartite user-item recommendation."""

        def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3) -> None:
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.num_layers = num_layers
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            nn.init.xavier_uniform_(self.user_embedding.weight)
            nn.init.xavier_uniform_(self.item_embedding.weight)

        def forward(self, norm_adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_layers = [embeddings]
            current = embeddings
            for _ in range(self.num_layers):
                current = torch.sparse.mm(norm_adj, current)
                all_layers.append(current)
            final = torch.stack(all_layers, dim=0).mean(dim=0)
            users, items = torch.split(final, [self.num_users, self.num_items], dim=0)
            return users, items

        def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
            user_embeddings, item_embeddings = self.forward(norm_adj)
            return (user_embeddings[user_ids] * item_embeddings[item_ids]).sum(dim=1)


    def build_normalized_adj(
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        num_users: int,
        num_items: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build D^-1/2 A D^-1/2 sparse adjacency for the user-item graph."""

        device = device or user_indices.device
        item_nodes = item_indices + num_users
        row = torch.cat([user_indices, item_nodes]).to(device)
        col = torch.cat([item_nodes, user_indices]).to(device)
        values = torch.ones(row.shape[0], dtype=torch.float32, device=device)
        node_count = num_users + num_items
        degree = torch.zeros(node_count, dtype=torch.float32, device=device)
        degree.index_add_(0, row, values)
        degree = torch.clamp(degree, min=1.0)
        norm_values = values * torch.pow(degree[row], -0.5) * torch.pow(degree[col], -0.5)
        indices = torch.stack([row, col], dim=0)
        return torch.sparse_coo_tensor(indices, norm_values, (node_count, node_count), device=device).coalesce()

else:

    class LightGCNModel:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("LightGCNModel requires torch. Install requirements-ml.txt.")


    def build_normalized_adj(*args, **kwargs):  # pragma: no cover - optional dependency
        raise ImportError("build_normalized_adj requires torch. Install requirements-ml.txt.")
