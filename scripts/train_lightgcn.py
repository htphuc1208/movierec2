from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k


@dataclass(frozen=True)
class LightGCNTrainingResult:
    train_loss: float
    val_loss: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    best_epoch: int


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit(
            "Torch is required for scripts/train_lightgcn.py. "
            "Install it with: pip install -r requirements-ml.txt"
        ) from exc
    return torch, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_id_maps(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
    movie_ids = movies["movieId"].astype(int).tolist()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


class BPRDataset:
    """Positive user-item pairs used by LightGCN with BPR negative sampling."""

    def __init__(self, df: pd.DataFrame, user_to_idx: dict[int, int], item_to_idx: dict[int, int], threshold: float, torch):
        filtered = df[(df["userId"].astype(int).isin(user_to_idx)) & (df["movieId"].astype(int).isin(item_to_idx))]
        pos_df = filtered[filtered["rating"] >= threshold]

        self.users = torch.tensor([user_to_idx[int(uid)] for uid in pos_df["userId"]], dtype=torch.long)
        self.pos_items = torch.tensor([item_to_idx[int(mid)] for mid in pos_df["movieId"]], dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx]


def build_negative_candidates(user_positive_items: dict[int, set[int]], num_items: int) -> dict[int, np.ndarray]:
    all_items = np.arange(num_items, dtype=np.int64)
    candidates: dict[int, np.ndarray] = {}
    for user_idx, positives in user_positive_items.items():
        positive_array = np.asarray(sorted(positives), dtype=np.int64)
        candidates[user_idx] = np.setdiff1d(all_items, positive_array, assume_unique=True)
    return candidates


def sample_negative_batch(
    users: Any,
    negative_candidates: dict[int, np.ndarray],
    num_items: int,
    torch,
) -> Any:
    sampled: list[int] = []
    for user_idx in users.detach().cpu().tolist():
        candidates = negative_candidates.get(int(user_idx))
        if candidates is None or len(candidates) == 0:
            sampled.append(int(np.random.randint(0, max(num_items, 1))))
        else:
            sampled.append(int(np.random.choice(candidates)))
    return torch.tensor(sampled, dtype=torch.long, device=users.device)


def evaluate_top_k(
    model,
    norm_adj,
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    top_k: int,
    positive_threshold: float,
    torch,
) -> dict[str, float]:
    model.eval()
    train_seen = train.groupby("userId")["movieId"].apply(lambda values: set(int(value) for value in values)).to_dict()
    relevant_by_user = (
        test.loc[test["rating"] >= positive_threshold]
        .groupby("userId")["movieId"]
        .apply(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    if not relevant_by_user:
        return {f"precision@{top_k}": 0.0, f"recall@{top_k}": 0.0, f"ndcg@{top_k}": 0.0, f"mrr@{top_k}": 0.0}

    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []

    with torch.no_grad():
        user_embeds, item_embeds = model(norm_adj)
        user_embeds = user_embeds.cpu().numpy()
        item_embeds = item_embeds.cpu().numpy()

    for user_id, relevant in relevant_by_user.items():
        if int(user_id) not in user_to_idx:
            continue
        user_idx = user_to_idx[int(user_id)]
        
        scores = user_embeds[user_idx] @ item_embeds.T
        
        for movie_id in train_seen.get(user_id, set()):
            item_idx = item_to_idx.get(int(movie_id))
            if item_idx is not None:
                scores[item_idx] = -np.inf
                
        ranked_indices = np.argsort(scores)[::-1]
        recommended = [idx_to_item[int(idx)] for idx in ranked_indices[:top_k] if np.isfinite(scores[int(idx)])]
        
        precision_values.append(precision_at_k(recommended, relevant, top_k))
        recall_values.append(recall_at_k(recommended, relevant, top_k))
        ndcg_values.append(ndcg_at_k(recommended, relevant, top_k))
        mrr_values.append(mrr_at_k(recommended, relevant, top_k))

    def avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        f"precision@{top_k}": avg(precision_values),
        f"recall@{top_k}": avg(recall_values),
        f"ndcg@{top_k}": avg(ndcg_values),
        f"mrr@{top_k}": avg(mrr_values),
    }


def export_recommender_artifact(
    model: Any,
    norm_adj: Any,
    output_dir: str | Path,
    idx_to_user: dict[int, int],
    idx_to_item: dict[int, int],
    dataset_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    positive_threshold: float,
    torch,
) -> Path:
    """Export final LightGCN embeddings into the lightweight API/UI format."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(norm_adj)
    user_embeddings_np = user_embeddings.detach().cpu().numpy().astype(np.float32)
    item_embeddings_np = item_embeddings.detach().cpu().numpy().astype(np.float32)
    user_ids = np.asarray([idx_to_user[idx] for idx in range(len(idx_to_user))], dtype=np.int64)
    movie_ids = np.asarray([idx_to_item[idx] for idx in range(len(idx_to_item))], dtype=np.int64)

    np.savez_compressed(
        path / "collaborative.npz",
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_embeddings=user_embeddings_np,
        item_embeddings=item_embeddings_np,
        global_mean=np.asarray([0.0], dtype=np.float32),
    )

    manifest = {
        "artifact_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "model_name": "lightgcn-pytorch",
        "model_source": "pytorch_lightgcn",
        "positive_threshold": positive_threshold,
        "weights": {"collaborative": 0.55, "content": 0.35, "popularity": 0.10},
        "collaborative": {
            "mode": "embedding",
            "engine": "torch-native",
            "factors": int(item_embeddings_np.shape[1]),
            "layers": int(config.get("layers", 0)),
        },
        "content": {"backend": "tfidf"},
        "files": {"collaborative": "collaborative.npz"},
        "metrics": {"test": metrics},
        "config": config,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def train(args: argparse.Namespace) -> LightGCNTrainingResult:
    torch, DataLoader = require_torch()
    from models.LightGCN import LightGCNModel, build_normalized_adj

    set_seed(args.seed)
    loader = MovieLensDataLoader(args.data_dir)
    bundle = loader.load()
    train_df, val_df, test_df = loader.train_val_test_split(bundle.ratings)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(bundle.movies, train_df)

    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    pos_df_filtered = train_df[(train_df["userId"].astype(int).isin(user_to_idx)) & (train_df["movieId"].astype(int).isin(item_to_idx))]
    pos_df_filtered = pos_df_filtered[pos_df_filtered["rating"] >= args.positive_threshold]

    train_pos_graph: dict[int, set[int]] = {}
    for row in pos_df_filtered.itertuples():
        u = user_to_idx[int(row.userId)]
        i = item_to_idx[int(row.movieId)]
        train_pos_graph.setdefault(u, set()).add(i)
    negative_candidates = build_negative_candidates(train_pos_graph, num_items)

    u_indices = [user_to_idx[int(uid)] for uid in pos_df_filtered["userId"]]
    i_indices = [item_to_idx[int(mid)] for mid in pos_df_filtered["movieId"]]
    u_tensor = torch.tensor(u_indices, dtype=torch.long)
    i_tensor = torch.tensor(i_indices, dtype=torch.long)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    norm_adj = build_normalized_adj(u_tensor, i_tensor, num_users, num_items, device=device)

    train_dataset = BPRDataset(train_df, user_to_idx, item_to_idx, args.positive_threshold, torch)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    val_dataset = BPRDataset(val_df, user_to_idx, item_to_idx, args.positive_threshold, torch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = LightGCNModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.factors,
        num_layers=args.layers,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = args.patience
    train_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        total_train_batches = 0

        for batch_users, batch_pos_items in train_loader:
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = sample_negative_batch(batch_users, negative_candidates, num_items, torch)

            optimizer.zero_grad()
            user_embeds, item_embeds = model(norm_adj)

            pos_scores = torch.sum(user_embeds[batch_users] * item_embeds[batch_pos_items], dim=1)
            neg_scores = torch.sum(user_embeds[batch_users] * item_embeds[batch_neg_items], dim=1)

            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            if args.l2_reg > 0:
                loss = loss + args.l2_reg * (
                    user_embeds[batch_users].pow(2).sum(dim=1)
                    + item_embeds[batch_pos_items].pow(2).sum(dim=1)
                    + item_embeds[batch_neg_items].pow(2).sum(dim=1)
                ).mean()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_train_loss += loss.item()
            total_train_batches += 1

        train_loss = epoch_train_loss / max(total_train_batches, 1)

        model.eval()
        epoch_val_loss = 0.0
        total_val_batches = 0

        if len(val_dataset) > 0:
            with torch.no_grad():
                user_embeds, item_embeds = model(norm_adj)
                for batch_users, batch_pos_items in val_loader:
                    batch_users = batch_users.to(device)
                    batch_pos_items = batch_pos_items.to(device)
                    batch_neg_items = sample_negative_batch(batch_users, negative_candidates, num_items, torch)

                    pos_scores = torch.sum(user_embeds[batch_users] * item_embeds[batch_pos_items], dim=1)
                    neg_scores = torch.sum(user_embeds[batch_users] * item_embeds[batch_neg_items], dim=1)

                    v_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
                    epoch_val_loss += v_loss.item()
                    total_val_batches += 1
            val_loss = epoch_val_loss / max(total_val_batches, 1)
        else:
            val_loss = train_loss

        if args.verbose:
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if args.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    top_k_metrics = evaluate_top_k(
        model, norm_adj, train_df, test_df, user_to_idx, item_to_idx, idx_to_item,
        args.top_k, args.positive_threshold, torch,
    )

    result = LightGCNTrainingResult(
        train_loss=train_loss,
        val_loss=best_val_loss,
        precision_at_k=top_k_metrics[f"precision@{args.top_k}"],
        recall_at_k=top_k_metrics[f"recall@{args.top_k}"],
        ndcg_at_k=top_k_metrics[f"ndcg@{args.top_k}"],
        mrr_at_k=top_k_metrics[f"mrr@{args.top_k}"],
        best_epoch=best_epoch,
    )

    if args.artifact_path:
        artifact_path = Path(args.artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "user_to_idx": user_to_idx,
                "item_to_idx": item_to_idx,
                "idx_to_user": idx_to_user,
                "idx_to_item": idx_to_item,
                "config": vars(args),
                "metrics": asdict(result),
            },
            artifact_path,
        )
        metrics_path = artifact_path.with_suffix(".metrics.json")
        metrics_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(f"saved artifact: {artifact_path}")
        print(f"saved metrics: {metrics_path}")

    if args.recommender_artifact_dir:
        dataset_name = args.dataset_name or Path(args.data_dir).resolve().name
        recommender_artifact_dir = export_recommender_artifact(
            model=model,
            norm_adj=norm_adj,
            output_dir=args.recommender_artifact_dir,
            idx_to_user=idx_to_user,
            idx_to_item=idx_to_item,
            dataset_name=dataset_name,
            config=vars(args),
            metrics=asdict(result),
            positive_threshold=args.positive_threshold,
            torch=torch,
        )
        print(f"saved recommender artifact: {recommender_artifact_dir}")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an independent pure LightGCN + BPR recommender.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--artifact-path", default="artifacts/lightgcn_baseline.pt")
    parser.add_argument("--recommender-artifact-dir", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = train(parse_args())
    print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH THUẦN LIGHTGCN ---")
    for name, value in asdict(result).items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
