#!/usr/bin/env python3
"""Train/export the hybrid recommender artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from recommender.config import PROJECT_ROOT
from recommender.data.movielens import (
    build_sparse_interaction_matrix,
    build_user_item_sets,
    filter_catalog_to_items,
    prepare_interactions,
    read_movielens,
)
from recommender.eval.metrics import evaluate_score_fn, minmax
from recommender.inference.artifacts import save_artifact_bundle
from recommender.models.learned_two_tower import LearnedTwoTowerRecommender
from recommender.models.svd import evaluate_svd_rmse, fit_svd_baseline
from recommender.models.two_tower import build_user_profiles, encode_item_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hybrid MovieLens recommender")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ml-latest-small")
    parser.add_argument("--enriched-catalog", type=Path, default=PROJECT_ROOT / "data" / "processed" / "movie_catalog_enriched.parquet")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--content-backend", choices=["sbert", "tfidf", "auto"], default="sbert")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--train-lightgcn", action="store_true")
    parser.add_argument("--train-two-tower", action="store_true", help="Train learned user/item two-tower on content embeddings")
    parser.add_argument("--lightgcn-dim", type=int, default=64)
    parser.add_argument("--lightgcn-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hybrid-grid-step", type=float, default=0.25)
    parser.add_argument(
        "--no-store-train-user-items",
        action="store_true",
        help="Do not store watched training items in hybrid_config.json. Useful for very large experiments.",
    )
    return parser.parse_args()


def ordered_catalog(raw_movies: pd.DataFrame, enriched_path: Path, item_mapping: dict[int, int]) -> pd.DataFrame:
    if enriched_path.exists():
        catalog_source = pd.read_parquet(enriched_path)
    else:
        catalog_source = raw_movies
    return filter_catalog_to_items(catalog_source, item_mapping)


def item_popularity(train: pd.DataFrame, num_items: int) -> np.ndarray:
    counts = np.bincount(train["item_idx"].to_numpy(dtype=np.int64), minlength=num_items).astype(np.float32)
    counts = np.log1p(counts)
    return minmax(counts)


def train_lightgcn_if_requested(args: argparse.Namespace, prepared, train_user_items):
    if not args.train_lightgcn:
        return None, None, {"enabled": False, "reason": "not requested"}
    try:
        import torch

        from recommender.models.lightgcn import LightGCNModel, build_normalized_adj, train_lightgcn_bpr
    except ImportError as exc:
        return None, None, {"enabled": False, "reason": str(exc)}

    edges = prepared.train[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
    adjacency = build_normalized_adj(prepared.num_users, prepared.num_items, edges, device=args.device)
    model = LightGCNModel(
        prepared.num_users,
        prepared.num_items,
        embedding_dim=args.lightgcn_dim,
        num_layers=args.lightgcn_layers,
        adjacency=adjacency,
    )
    losses = train_lightgcn_bpr(
        model,
        train_user_items,
        prepared.num_items,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate()
    return (
        user_embeddings.detach().cpu().numpy().astype(np.float32),
        item_embeddings.detach().cpu().numpy().astype(np.float32),
        {"enabled": True, "device": args.device, "losses": losses},
    )


def train_two_tower_if_requested(
    args: argparse.Namespace,
    prepared,
    train_user_items: dict[int, set[int]],
    content_embeddings: np.ndarray,
):
    if not args.train_two_tower:
        return None, None, {"enabled": False, "reason": "not requested"}
    dataset = SimpleNamespace(
        train=prepared.train,
        train_user_items=train_user_items,
        num_users=prepared.num_users,
        num_items=prepared.num_items,
        content_embeddings=content_embeddings,
    )
    try:
        model = LearnedTwoTowerRecommender(
            embedding_dim=args.lightgcn_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=42,
        ).fit(dataset)
    except Exception as exc:
        return None, None, {"enabled": False, "reason": str(exc)}
    return model.user_embeddings_, model.item_embeddings_, {"enabled": True, **model.metadata}


def score_factory(
    user_profiles: np.ndarray,
    content_embeddings: np.ndarray,
    popularity: np.ndarray,
    lightgcn_user_embeddings: np.ndarray | None,
    lightgcn_item_embeddings: np.ndarray | None,
    two_tower_user_embeddings: np.ndarray | None,
    two_tower_item_embeddings: np.ndarray | None,
    weights: dict[str, float],
):
    def score_fn(batch_users: np.ndarray) -> np.ndarray:
        content = user_profiles[batch_users] @ content_embeddings.T
        scores = weights.get("content_weight", 0.0) * minmax(content, axis=1)
        pop = np.broadcast_to(popularity[None, :], content.shape)
        scores = scores + weights.get("popularity_weight", 0.0) * pop
        cf_weight = weights.get("cf_weight", 0.0)
        if lightgcn_user_embeddings is not None and lightgcn_item_embeddings is not None and cf_weight:
            cf = lightgcn_user_embeddings[batch_users] @ lightgcn_item_embeddings.T
            scores = scores + cf_weight * minmax(cf, axis=1)
        two_tower_weight = weights.get("two_tower_weight", 0.0)
        if two_tower_user_embeddings is not None and two_tower_item_embeddings is not None and two_tower_weight:
            two_tower = two_tower_user_embeddings[batch_users] @ two_tower_item_embeddings.T
            scores = scores + two_tower_weight * minmax(two_tower, axis=1)
        return scores.astype(np.float32)

    return score_fn


def tune_weights(
    num_users: int,
    num_items: int,
    user_profiles: np.ndarray,
    content_embeddings: np.ndarray,
    popularity: np.ndarray,
    lightgcn_user_embeddings: np.ndarray | None,
    lightgcn_item_embeddings: np.ndarray | None,
    two_tower_user_embeddings: np.ndarray | None,
    two_tower_item_embeddings: np.ndarray | None,
    train_user_items: dict[int, set[int]],
    val_user_items: dict[int, set[int]],
    k: int,
    grid_step: float,
) -> tuple[dict[str, float], dict[str, float]]:
    active_weights = ["content_weight", "popularity_weight"]
    if lightgcn_user_embeddings is not None and lightgcn_item_embeddings is not None:
        active_weights.insert(0, "cf_weight")
    if two_tower_user_embeddings is not None and two_tower_item_embeddings is not None:
        insert_at = 1 if "cf_weight" in active_weights else 0
        active_weights.insert(insert_at, "two_tower_weight")

    def as_weight_dict(values: np.ndarray) -> dict[str, float]:
        weights = {"cf_weight": 0.0, "two_tower_weight": 0.0, "content_weight": 0.0, "popularity_weight": 0.0}
        for name, value in zip(active_weights, values, strict=False):
            weights[name] = float(value)
        return weights

    if not val_user_items:
        weights = as_weight_dict(np.ones(len(active_weights), dtype=np.float32) / max(1, len(active_weights)))
        return weights, {}

    candidates = _weight_grid(len(active_weights), grid_step)
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] = {}
    best_ndcg = -1.0
    for candidate in candidates:
        weights = as_weight_dict(candidate)
        metrics = evaluate_score_fn(
            num_users,
            num_items,
            score_factory(
                user_profiles,
                content_embeddings,
                popularity,
                lightgcn_user_embeddings,
                lightgcn_item_embeddings,
                two_tower_user_embeddings,
                two_tower_item_embeddings,
                weights,
            ),
            train_user_items,
            val_user_items,
            k=k,
        )
        ndcg = metrics.get(f"ndcg@{k}", 0.0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_metrics = metrics
            best_weights = weights
    return best_weights or as_weight_dict(np.ones(len(active_weights), dtype=np.float32) / max(1, len(active_weights))), best_metrics


def _weight_grid(component_count: int, step: float) -> list[np.ndarray]:
    if component_count <= 1:
        return [np.ones(1, dtype=np.float32)]
    steps = max(1, int(round(1.0 / max(step, 1e-6))))
    candidates: list[np.ndarray] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            candidates.append(np.asarray([*prefix, remaining], dtype=np.float32) / steps)
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots - 1)

    build([], steps, component_count)
    return candidates


def main() -> None:
    args = parse_args()
    _validate_device(args.device)
    data = read_movielens(args.raw_dir)
    prepared = prepare_interactions(data.ratings, min_rating=args.min_rating)
    catalog = ordered_catalog(data.movies, args.enriched_catalog, prepared.item_mapping)

    print(f"Users={prepared.num_users} Items={prepared.num_items} Train={len(prepared.train)} Val={len(prepared.val)} Test={len(prepared.test)}")
    content_embeddings = encode_item_texts(
        catalog,
        backend=args.content_backend,
        model_name=args.sbert_model,
        device=args.device if args.content_backend in {"sbert", "auto"} else None,
    )
    user_profiles = build_user_profiles(prepared.train, content_embeddings, prepared.num_users)
    popularity = item_popularity(prepared.train, prepared.num_items)

    train_user_items = build_user_item_sets(prepared.train)
    val_user_items = build_user_item_sets(prepared.val)
    test_user_items = build_user_item_sets(prepared.test)
    train_matrix = build_sparse_interaction_matrix(prepared.train, prepared.num_users, prepared.num_items)

    try:
        svd_user_factors, svd_item_factors, _ = fit_svd_baseline(prepared.train, prepared.num_users, prepared.num_items)
        svd_rmse = evaluate_svd_rmse(svd_user_factors, svd_item_factors, prepared.test)
    except Exception as exc:
        svd_rmse = 0.0
        print(f"SVD baseline skipped: {exc}")

    lightgcn_user, lightgcn_item, lightgcn_info = train_lightgcn_if_requested(args, prepared, train_user_items)
    two_tower_user, two_tower_item, two_tower_info = train_two_tower_if_requested(args, prepared, train_user_items, content_embeddings)
    weights, val_metrics = tune_weights(
        prepared.num_users,
        prepared.num_items,
        user_profiles,
        content_embeddings,
        popularity,
        lightgcn_user,
        lightgcn_item,
        two_tower_user,
        two_tower_item,
        train_user_items,
        val_user_items,
        args.k,
        args.hybrid_grid_step,
    )
    test_metrics = evaluate_score_fn(
        prepared.num_users,
        prepared.num_items,
        score_factory(
            user_profiles,
            content_embeddings,
            popularity,
            lightgcn_user,
            lightgcn_item,
            two_tower_user,
            two_tower_item,
            weights,
        ),
        train_user_items,
        test_user_items,
        k=args.k,
    )

    metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "svd_rmse": svd_rmse,
        "lightgcn": lightgcn_info,
        "two_tower": two_tower_info,
        "split_stats": {
            "train_interactions": int(len(prepared.train)),
            "val_interactions": int(len(prepared.val)),
            "test_interactions": int(len(prepared.test)),
            "train_items_nonzero": int(train_matrix.getnnz(axis=0).astype(bool).sum()),
        },
    }
    hybrid_config = {
        **weights,
        "model_type": "hybrid_pdf_clean" if args.train_two_tower else "hybrid_weighted",
        "k": args.k,
        "min_rating": args.min_rating,
        "content_backend": args.content_backend,
        "sbert_model": args.sbert_model,
        "hybrid_grid_step": args.hybrid_grid_step,
    }
    if not args.no_store_train_user_items:
        hybrid_config["train_user_items"] = {str(user): sorted(items) for user, items in train_user_items.items()}

    save_artifact_bundle(
        args.artifacts_dir,
        catalog=catalog,
        user_mapping=prepared.user_mapping,
        item_mapping=prepared.item_mapping,
        content_embeddings=content_embeddings,
        user_profiles=user_profiles,
        item_popularity=popularity,
        metrics=metrics,
        hybrid_config=hybrid_config,
        lightgcn_user_embeddings=lightgcn_user,
        lightgcn_item_embeddings=lightgcn_item,
        two_tower_user_embeddings=two_tower_user,
        two_tower_item_embeddings=two_tower_item,
    )

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with (args.artifacts_dir / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"metrics": metrics, "hybrid_config": hybrid_config}, fh, ensure_ascii=False, indent=2)
    print(f"Saved artifacts to {args.artifacts_dir}")
    print(json.dumps({"metrics": metrics, "hybrid_config": hybrid_config}, ensure_ascii=False, indent=2))


def _validate_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("device=cuda was requested, but torch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but torch.cuda.is_available() is False")


if __name__ == "__main__":
    main()
