from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse


@dataclass(frozen=True)
class SVDTrainingResult:
    train_rmse: float
    val_rmse: float
    test_rmse: float
    test_mae: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    best_epoch: int


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "Torch is required for scripts/train_svd.py. "
            "Install it with: pip install -r requirements-ml.txt"
        ) from exc
    return torch, DataLoader, TensorDataset


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


def ratings_to_tensors(ratings: pd.DataFrame, user_to_idx: dict[int, int], item_to_idx: dict[int, int], torch) -> tuple[Any, Any, Any]:
    filtered = ratings[
        ratings["userId"].astype(int).isin(user_to_idx)
        & ratings["movieId"].astype(int).isin(item_to_idx)
    ]
    users = torch.tensor([user_to_idx[int(user_id)] for user_id in filtered["userId"]], dtype=torch.long)
    items = torch.tensor([item_to_idx[int(movie_id)] for movie_id in filtered["movieId"]], dtype=torch.long)
    labels = torch.tensor(filtered["rating"].astype(float).to_numpy(), dtype=torch.float32)
    return users, items, labels


def predict_frame(model, frame: pd.DataFrame, user_to_idx: dict[int, int], item_to_idx: dict[int, int], batch_size: int, device, torch) -> tuple[list[float], list[float]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    rows = frame[
        frame["userId"].astype(int).isin(user_to_idx)
        & frame["movieId"].astype(int).isin(item_to_idx)
    ]
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows.iloc[start : start + batch_size]
            users = torch.tensor([user_to_idx[int(user_id)] for user_id in batch["userId"]], dtype=torch.long, device=device)
            items = torch.tensor([item_to_idx[int(movie_id)] for movie_id in batch["movieId"]], dtype=torch.long, device=device)
            preds = model(users, items).clamp(0.5, 5.0).detach().cpu().numpy().tolist()
            y_pred.extend(float(value) for value in preds)
            y_true.extend(float(value) for value in batch["rating"])
    return y_true, y_pred


def score_all_items_for_user(model, user_idx: int, num_items: int, batch_size: int, device, torch) -> np.ndarray:
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            item_ids = torch.arange(start, min(start + batch_size, num_items), dtype=torch.long, device=device)
            user_ids = torch.full_like(item_ids, fill_value=user_idx)
            preds = model(user_ids, item_ids).detach().cpu().numpy().tolist()
            scores.extend(float(value) for value in preds)
    return np.asarray(scores, dtype=np.float32)


def evaluate_top_k(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    top_k: int,
    positive_threshold: float,
    batch_size: int,
    device,
    torch,
) -> dict[str, float]:
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

    for user_id, relevant in relevant_by_user.items():
        if int(user_id) not in user_to_idx:
            continue
        user_idx = user_to_idx[int(user_id)]
        scores = score_all_items_for_user(model, user_idx, len(item_to_idx), batch_size, device, torch)
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


def train(args: argparse.Namespace) -> SVDTrainingResult:
    torch, DataLoader, TensorDataset = require_torch()
    from models.SVD import SVDModel

    set_seed(args.seed)
    loader = MovieLensDataLoader(args.data_dir)
    bundle = loader.load()
    train_df, val_df, test_df = loader.train_val_test_split(bundle.ratings)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(bundle.movies, train_df)

    train_users, train_items, train_labels = ratings_to_tensors(train_df, user_to_idx, item_to_idx, torch)
    val_users, val_items, val_labels = ratings_to_tensors(val_df, user_to_idx, item_to_idx, torch)
    train_dataset = TensorDataset(train_users, train_items, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    global_mean = float(train_df["rating"].mean())
    model = SVDModel(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        embedding_dim=args.factors,
        global_mean=global_mean,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for users, items, labels in train_loader:
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * labels.numel()
            train_count += labels.numel()

        with torch.no_grad():
            model.eval()
            val_preds = model(val_users.to(device), val_items.to(device)).clamp(0.5, 5.0)
            val_rmse = float(torch.sqrt(torch.mean((val_preds - val_labels.to(device)) ** 2)).item()) if len(val_labels) else 0.0

        train_rmse = float(np.sqrt(train_loss / max(train_count, 1)))
        if args.verbose:
            print(f"epoch={epoch:03d} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_true, train_pred = predict_frame(model, train_df, user_to_idx, item_to_idx, args.batch_size, device, torch)
    val_true, val_pred = predict_frame(model, val_df, user_to_idx, item_to_idx, args.batch_size, device, torch)
    test_true, test_pred = predict_frame(model, test_df, user_to_idx, item_to_idx, args.batch_size, device, torch)
    top_k_metrics = evaluate_top_k(
        model,
        train_df,
        test_df,
        user_to_idx,
        item_to_idx,
        idx_to_item,
        args.top_k,
        args.positive_threshold,
        args.batch_size,
        device,
        torch,
    )

    result = SVDTrainingResult(
        train_rmse=rmse(train_true, train_pred),
        val_rmse=rmse(val_true, val_pred),
        test_rmse=rmse(test_true, test_pred),
        test_mae=float(np.mean(np.abs(np.asarray(test_true, dtype=np.float32) - np.asarray(test_pred, dtype=np.float32)))) if test_true else 0.0,
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

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an independent biased Funk-SVD baseline.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--artifact-path", default="artifacts/svd_baseline.pt")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = train(parse_args())
    for name, value in asdict(result).items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
