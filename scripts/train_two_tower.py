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
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from models import MetadataEncoder, TwoTowerModel


@dataclass(frozen=True)
class TwoTowerTrainingResult:
    train_loss: float
    val_loss: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    mrr_at_k: float
    all_precision_at_k: float
    all_recall_at_k: float
    all_ndcg_at_k: float
    all_mrr_at_k: float
    cold_precision_at_k: float
    cold_recall_at_k: float
    cold_ndcg_at_k: float
    cold_mrr_at_k: float
    train_item_count: int
    catalog_item_count: int
    cold_test_interactions: int
    warm_test_interactions: int
    cold_positive_interactions: int
    best_epoch: int


class PairDataset:
    def __init__(self, pairs: list[tuple[int, int]], torch) -> None:
        self.users = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
        self.items = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise SystemExit("Torch is required for scripts/train_two_tower.py. Install requirements-ml.txt.") from exc
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


def encode_tfidf_vectors(movies: pd.DataFrame, tags: pd.DataFrame | None, max_features: int) -> np.ndarray:
    normalised = MetadataEncoder._normalise_movies(movies, tags)
    texts = MetadataEncoder._movie_texts(normalised)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=max(2, int(max_features)),
    )
    matrix = vectorizer.fit_transform(texts).astype(np.float32)
    return l2_normalise(matrix.toarray().astype(np.float32))


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_id_maps(movies: pd.DataFrame, train: pd.DataFrame) -> tuple[dict[int, int], dict[int, int], dict[int, int], dict[int, int]]:
    user_ids = sorted(train["userId"].astype(int).unique().tolist())
    movie_ids = movies["movieId"].astype(int).tolist()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_item = {idx: movie_id for movie_id, idx in item_to_idx.items()}
    return user_to_idx, item_to_idx, idx_to_user, idx_to_item


def positive_pairs(
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    positive_threshold: float,
) -> list[tuple[int, int]]:
    positives = frame.loc[pd.to_numeric(frame["rating"], errors="coerce") >= positive_threshold]
    pairs: list[tuple[int, int]] = []
    for row in positives.itertuples():
        user_idx = user_to_idx.get(int(row.userId))
        item_idx = item_to_idx.get(int(row.movieId))
        if user_idx is not None and item_idx is not None:
            pairs.append((user_idx, item_idx))
    return pairs


def build_user_features(
    train: pd.DataFrame,
    item_vectors: np.ndarray,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    positive_threshold: float,
) -> np.ndarray:
    features = np.zeros((len(user_to_idx), item_vectors.shape[1]), dtype=np.float32)
    for user_id, group in train.groupby("userId"):
        user_idx = user_to_idx.get(int(user_id))
        if user_idx is None:
            continue
        positive_ids = group.loc[group["rating"] >= positive_threshold, "movieId"].astype(int).tolist()
        movie_ids = positive_ids or group["movieId"].astype(int).tolist()
        item_indices = [item_to_idx[movie_id] for movie_id in movie_ids if movie_id in item_to_idx]
        if item_indices:
            features[user_idx] = item_vectors[item_indices].mean(axis=0)
    return l2_normalise(features)


def build_negative_candidates(pairs: list[tuple[int, int]], num_items: int) -> dict[int, np.ndarray]:
    all_items = np.arange(num_items, dtype=np.int64)
    positives: dict[int, set[int]] = {}
    for user_idx, item_idx in pairs:
        positives.setdefault(user_idx, set()).add(item_idx)
    return {user_idx: np.setdiff1d(all_items, np.asarray(sorted(items), dtype=np.int64)) for user_idx, items in positives.items()}


def sample_negative_batch(users: Any, candidates: dict[int, np.ndarray], num_items: int, torch) -> Any:
    sampled: list[int] = []
    for user_idx in users.detach().cpu().tolist():
        available = candidates.get(int(user_idx))
        if available is None or len(available) == 0:
            sampled.append(int(np.random.randint(0, max(num_items, 1))))
        else:
            sampled.append(int(np.random.choice(available)))
    return torch.tensor(sampled, dtype=torch.long, device=users.device)


def evaluate_top_k(
    model,
    user_features: np.ndarray,
    item_features: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    idx_to_item: dict[int, int],
    top_k: int,
    positive_threshold: float,
    device,
    torch,
) -> dict[str, float]:
    relevant_by_user = (
        test.loc[test["rating"] >= positive_threshold]
        .groupby("userId")["movieId"]
        .apply(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    if not relevant_by_user:
        return {f"precision@{top_k}": 0.0, f"recall@{top_k}": 0.0, f"ndcg@{top_k}": 0.0, f"mrr@{top_k}": 0.0}

    train_seen = train.groupby("userId")["movieId"].apply(lambda values: set(int(value) for value in values)).to_dict()
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor(user_features, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(item_features, dtype=torch.float32, device=device)
        user_embeddings = torch.nn.functional.normalize(model.user_tower(user_tensor), p=2, dim=1).cpu().numpy()
        item_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor), p=2, dim=1).cpu().numpy()

    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []
    for user_id, relevant in relevant_by_user.items():
        user_idx = user_to_idx.get(int(user_id))
        if user_idx is None:
            continue
        scores = user_embeddings[user_idx] @ item_embeddings.T
        for movie_id in train_seen.get(user_id, set()):
            item_idx = item_to_idx.get(int(movie_id))
            if item_idx is not None:
                scores[item_idx] = -np.inf
        ranked = np.argsort(scores)[::-1]
        recommended = [idx_to_item[int(idx)] for idx in ranked[:top_k] if np.isfinite(scores[int(idx)])]
        precision_values.append(precision_at_k(recommended, relevant, top_k))
        recall_values.append(recall_at_k(recommended, relevant, top_k))
        ndcg_values.append(ndcg_at_k(recommended, relevant, top_k))
        mrr_values.append(mrr_at_k(recommended, relevant, top_k))

    return {
        f"precision@{top_k}": mean(precision_values),
        f"recall@{top_k}": mean(recall_values),
        f"ndcg@{top_k}": mean(ndcg_values),
        f"mrr@{top_k}": mean(mrr_values),
    }


def export_recommender_artifact(
    model: Any,
    user_features: np.ndarray,
    item_features: np.ndarray,
    output_dir: str | Path,
    idx_to_user: dict[int, int],
    idx_to_item: dict[int, int],
    dataset_name: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    positive_threshold: float,
    device,
    torch,
    weights: dict[str, float] | None = None,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor(user_features, dtype=torch.float32, device=device)
        item_tensor = torch.tensor(item_features, dtype=torch.float32, device=device)
        user_embeddings = torch.nn.functional.normalize(model.user_tower(user_tensor), p=2, dim=1).cpu().numpy().astype(np.float32)
        item_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor), p=2, dim=1).cpu().numpy().astype(np.float32)

    np.savez_compressed(
        path / "collaborative.npz",
        user_ids=np.asarray([idx_to_user[idx] for idx in range(len(idx_to_user))], dtype=np.int64),
        movie_ids=np.asarray([idx_to_item[idx] for idx in range(len(idx_to_item))], dtype=np.int64),
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        global_mean=np.asarray([0.0], dtype=np.float32),
    )
    manifest_weights = weights or {"collaborative": 0.55, "content": 0.35, "popularity": 0.10}
    manifest = {
        "artifact_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "model_name": "two-tower-tfidf",
        "model_source": "pytorch_two_tower",
        "positive_threshold": positive_threshold,
        "weights": manifest_weights,
        "collaborative": {"mode": "embedding", "engine": "torch-native", "factors": int(user_embeddings.shape[1])},
        "content": {"backend": "tfidf"},
        "files": {"collaborative": "collaborative.npz"},
        "metrics": {"test": metrics},
        "config": config,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def train(args: argparse.Namespace) -> TwoTowerTrainingResult:
    torch, DataLoader = require_torch()
    set_seed(args.seed)

    loader = MovieLensDataLoader(args.data_dir)
    bundle = loader.load()
    train_df, val_df, test_df = loader.train_val_test_split(bundle.ratings)
    warm_test_df, cold_test_df = loader.split_warm_cold_items(train_df, test_df)

    if args.content_backend != "tfidf":
        raise SystemExit("scripts/train_two_tower.py currently supports --content-backend tfidf.")
    item_vectors = encode_tfidf_vectors(bundle.movies, bundle.tags, args.max_feature_dim)
    user_to_idx, item_to_idx, idx_to_user, idx_to_item = build_id_maps(bundle.movies, train_df)
    user_features = build_user_features(train_df, item_vectors, user_to_idx, item_to_idx, args.positive_threshold)
    train_pairs = positive_pairs(train_df, user_to_idx, item_to_idx, args.positive_threshold)
    val_pairs = positive_pairs(val_df, user_to_idx, item_to_idx, args.positive_threshold)
    if not train_pairs:
        raise SystemExit("No positive training pairs for TwoTower.")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TwoTowerModel(
        input_dim=item_vectors.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = PairDataset(train_pairs, torch)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_dataset = PairDataset(val_pairs, torch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    negative_candidates = build_negative_candidates(train_pairs, len(item_to_idx))
    # Build separate validation negative candidates excluding both train and val positives
    val_positive_set: dict[int, set[int]] = {}
    for user_idx, item_idx in train_pairs:
        val_positive_set.setdefault(user_idx, set()).add(item_idx)
    for user_idx, item_idx in val_pairs:
        val_positive_set.setdefault(user_idx, set()).add(item_idx)
    val_negative_candidates = {user_idx: np.setdiff1d(np.arange(len(item_to_idx), dtype=np.int64), np.asarray(sorted(items), dtype=np.int64)) for user_idx, items in val_positive_set.items()}

    user_tensor = torch.tensor(user_features, dtype=torch.float32, device=device)
    item_tensor = torch.tensor(item_vectors, dtype=torch.float32, device=device)
    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_left = args.patience
    train_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for users, pos_items in train_loader:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = sample_negative_batch(users, negative_candidates, len(item_to_idx), torch)
            optimizer.zero_grad()
            user_embeddings = torch.nn.functional.normalize(model.user_tower(user_tensor[users]), p=2, dim=1)
            pos_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor[pos_items]), p=2, dim=1)
            neg_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor[neg_items]), p=2, dim=1)
            pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
            neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
            loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        train_loss = mean(losses)
        val_loss = train_loss
        if len(val_dataset) > 0:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for users, pos_items in val_loader:
                    users = users.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = sample_negative_batch(users, val_negative_candidates, len(item_to_idx), torch)
                    user_embeddings = torch.nn.functional.normalize(model.user_tower(user_tensor[users]), p=2, dim=1)
                    pos_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor[pos_items]), p=2, dim=1)
                    neg_embeddings = torch.nn.functional.normalize(model.item_tower(item_tensor[neg_items]), p=2, dim=1)
                    pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
                    neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
                    val_losses.append(float((-torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))).item()))
            val_loss = mean(val_losses)
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
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    warm_metrics = evaluate_top_k(model, user_features, item_vectors, train_df, warm_test_df, user_to_idx, item_to_idx, idx_to_item, args.top_k, args.positive_threshold, device, torch)
    all_metrics = evaluate_top_k(model, user_features, item_vectors, train_df, test_df, user_to_idx, item_to_idx, idx_to_item, args.top_k, args.positive_threshold, device, torch)
    cold_metrics = evaluate_top_k(model, user_features, item_vectors, train_df, cold_test_df, user_to_idx, item_to_idx, idx_to_item, args.top_k, args.positive_threshold, device, torch)
    result = TwoTowerTrainingResult(
        train_loss=train_loss,
        val_loss=best_val_loss,
        precision_at_k=warm_metrics[f"precision@{args.top_k}"],
        recall_at_k=warm_metrics[f"recall@{args.top_k}"],
        ndcg_at_k=warm_metrics[f"ndcg@{args.top_k}"],
        mrr_at_k=warm_metrics[f"mrr@{args.top_k}"],
        all_precision_at_k=all_metrics[f"precision@{args.top_k}"],
        all_recall_at_k=all_metrics[f"recall@{args.top_k}"],
        all_ndcg_at_k=all_metrics[f"ndcg@{args.top_k}"],
        all_mrr_at_k=all_metrics[f"mrr@{args.top_k}"],
        cold_precision_at_k=cold_metrics[f"precision@{args.top_k}"],
        cold_recall_at_k=cold_metrics[f"recall@{args.top_k}"],
        cold_ndcg_at_k=cold_metrics[f"ndcg@{args.top_k}"],
        cold_mrr_at_k=cold_metrics[f"mrr@{args.top_k}"],
        train_item_count=int(train_df["movieId"].nunique()),
        catalog_item_count=int(bundle.movies["movieId"].nunique()),
        cold_test_interactions=len(cold_test_df),
        warm_test_interactions=len(warm_test_df),
        cold_positive_interactions=int((cold_test_df["rating"] >= args.positive_threshold).sum()),
        best_epoch=best_epoch,
    )

    if args.artifact_path:
        artifact_path = Path(args.artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "metrics": asdict(result)}, artifact_path)
        artifact_path.with_suffix(".metrics.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(f"saved artifact: {artifact_path}")

    if args.recommender_artifact_dir:
        dataset_name = args.dataset_name or Path(args.data_dir).resolve().name
        output = export_recommender_artifact(
            model,
            user_features,
            item_vectors,
            args.recommender_artifact_dir,
            idx_to_user,
            idx_to_item,
            dataset_name,
            vars(args),
            asdict(result),
            args.positive_threshold,
            device,
            torch,
            weights={"collaborative": getattr(args, 'cf_weight', 0.55), "content": getattr(args, 'content_weight', 0.35), "popularity": getattr(args, 'popularity_weight', 0.10)},
        )
        print(f"saved recommender artifact: {output}")
    return result


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF metadata TwoTower recommender with BPR loss.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--artifact-path", default="artifacts/two_tower.pt")
    parser.add_argument("--recommender-artifact-dir", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--content-backend", choices=["tfidf"], default="tfidf")
    parser.add_argument("--max-feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--cf-weight", type=float, default=0.55)
    parser.add_argument("--content-weight", type=float, default=0.35)
    parser.add_argument("--popularity-weight", type=float, default=0.10)
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
