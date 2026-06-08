#!/usr/bin/env python3
"""Train/export the strongest Letterboxd-oriented hybrid ranker artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config import PROJECT_ROOT
from recommender.eval.metrics import evaluate_score_fn, minmax
from recommender.experiments.comparison import ComparisonConfig, evaluate_slice_metrics, load_experiment_dataset
from recommender.inference.artifacts import save_artifact_bundle
from recommender.models.baselines import (
    ContentAverageRecommender,
    EASERecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    SVDRankingRecommender,
    UserKNNRecommender,
)
from recommender.models.learned_two_tower import LearnedTwoTowerRecommender
from recommender.models.matrix_factorization import ImplicitALSRecommender, LightFMWARPRecommender, LightGCNRecommender
from recommender.models.rankers import StrongHybridRankerRecommender
from recommender.models.two_tower import build_user_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/export strongest hybrid ranker artifacts")
    parser.add_argument("--dataset", choices=["letterboxd", "movielens"], default="letterboxd")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "letterboxd")
    parser.add_argument("--enriched-catalog", type=Path, default=PROJECT_ROOT / "data" / "processed" / "letterboxd" / "movie_catalog_enriched.parquet")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "letterboxd_strong")
    parser.add_argument("--content-backend", choices=["sbert", "tfidf", "auto"], default="sbert")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--ranker", choices=["auto", "lightgbm", "sgd"], default="auto")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lightgcn-dim", type=int, default=128)
    parser.add_argument("--lightgcn-layers", type=int, default=3)
    parser.add_argument("--lightgcn-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-ease-items", type=int, default=5000)
    parser.add_argument("--max-ranker-samples", type=int, default=500_000)
    parser.add_argument("--embedding-cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "recommender" / "content_embeddings")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ComparisonConfig(
        k=args.k,
        batch_size=args.batch_size,
        models="full",
        content_backend=args.content_backend,
        sbert_model=args.sbert_model,
        min_rating=args.min_rating,
        epochs=args.lightgcn_epochs,
        device=args.device,
        mf_dim=args.lightgcn_dim,
        max_ease_items=args.max_ease_items,
        max_ranker_samples=args.max_ranker_samples,
        preset="letterboxd-strong",
        embedding_cache_dir=args.embedding_cache_dir,
        seed=args.seed,
    )
    dataset = load_experiment_dataset(args.dataset, args.raw_dir, args.enriched_catalog, config)
    components: list[Any] = []
    component_rows: list[dict[str, Any]] = []

    candidates: list[Any] = [
        PopularityRecommender(name="popularity_only"),
        ItemKNNRecommender(top_k=100),
        UserKNNRecommender(top_k=100),
        SVDRankingRecommender(n_components=128, random_state=args.seed),
        EASERecommender(max_items=args.max_ease_items),
        ContentAverageRecommender(name=f"{args.content_backend}_only", embedding_attr="content_embeddings"),
        LightGCNRecommender(
            embedding_dim=args.lightgcn_dim,
            num_layers=args.lightgcn_layers,
            epochs=args.lightgcn_epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
        LearnedTwoTowerRecommender(
            embedding_dim=args.lightgcn_dim,
            epochs=args.lightgcn_epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
        ImplicitALSRecommender(factors=args.lightgcn_dim, iterations=max(10, args.lightgcn_epochs // 5)),
        LightFMWARPRecommender(no_components=args.lightgcn_dim, epochs=max(10, args.lightgcn_epochs // 5), random_state=args.seed),
    ]

    for model in candidates:
        try:
            fitted = model.fit(dataset)
            metrics = _evaluate_model(dataset, fitted, config)
            components.append(fitted)
            component_rows.append({"model": fitted.name, "status": "ok", "metrics": metrics, "metadata": getattr(fitted, "metadata", {})})
            print(f"OK {fitted.name}: ndcg@{args.k}={metrics.get(f'ndcg@{args.k}', 0.0):.4f}")
        except Exception as exc:
            component_rows.append({"model": model.name, "status": "skipped_or_failed", "error": str(exc), "metadata": {}})
            print(f"SKIP {model.name}: {exc}")

    if len(components) < 2:
        raise RuntimeError("Need at least two fitted components for strong ranker")

    ranker = StrongHybridRankerRecommender(
        components=components,
        include_popularity=True,
        max_train_samples=args.max_ranker_samples,
        seed=args.seed,
        ranker=args.ranker,
        name="hybrid_strong_ranker",
    ).fit(dataset)
    ranker_metrics = _evaluate_model(dataset, ranker, config)
    component_rows.append({"model": ranker.name, "status": "ok", "metrics": ranker_metrics, "metadata": ranker.metadata})

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(ranker, args.artifacts_dir / "ranker.joblib")
    except Exception as exc:
        print(f"Could not persist ranker.joblib: {exc}")

    popularity = _item_popularity(dataset)
    user_profiles = build_user_profiles(dataset.train, dataset.content_embeddings, dataset.num_users)
    lightgcn = next((component for component in components if component.name == "lightgcn_only"), None)
    two_tower = next((component for component in components if component.name == "learned_two_tower"), None)
    hybrid_config = {
        "model_type": "strong_ranker",
        "ranker_path": "ranker.joblib",
        "ranker_metadata": ranker.metadata,
        "content_backend": args.content_backend,
        "sbert_model": args.sbert_model,
        "k": args.k,
        "min_rating": args.min_rating,
        "train_user_items": {str(user): sorted(items) for user, items in dataset.train_user_items.items()},
    }
    save_artifact_bundle(
        args.artifacts_dir,
        catalog=dataset.catalog,
        user_mapping=dataset.user_mapping,
        item_mapping=dataset.item_mapping,
        content_embeddings=dataset.content_embeddings,
        user_profiles=user_profiles,
        item_popularity=popularity,
        metrics={"test": ranker_metrics, "components": component_rows, "split_stats": dataset.split_stats},
        hybrid_config=hybrid_config,
        lightgcn_user_embeddings=getattr(lightgcn, "user_embeddings_", None),
        lightgcn_item_embeddings=getattr(lightgcn, "item_embeddings_", None),
        two_tower_user_embeddings=getattr(two_tower, "user_embeddings_", None),
        two_tower_item_embeddings=getattr(two_tower, "item_embeddings_", None),
    )
    (args.artifacts_dir / "component_score_config.json").write_text(
        json.dumps({"components": component_rows, "ranker": ranker.metadata, "split_stats": dataset.split_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ranker_metrics": ranker_metrics, "ranker_metadata": ranker.metadata}, ensure_ascii=False, indent=2))


def _evaluate_model(dataset, model, config: ComparisonConfig) -> dict[str, float]:
    metrics = evaluate_score_fn(
        dataset.num_users,
        dataset.num_items,
        lambda users: model.score_users(users),
        dataset.train_user_items,
        dataset.test_user_items,
        k=config.k,
        batch_size=config.batch_size,
    )
    metrics.update(evaluate_slice_metrics(dataset, lambda users: model.score_users(users), config))
    return metrics


def _item_popularity(dataset) -> np.ndarray:
    counts = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
    return minmax(np.log1p(counts))


if __name__ == "__main__":
    main()
