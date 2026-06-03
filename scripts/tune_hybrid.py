from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse
from models import HybridMovieRecommender


def dataset_name_from_dir(data_dir: str | Path) -> str:
    return Path(data_dir).resolve().name.replace("_", "-")


def candidate_weights() -> list[tuple[float, float, float]]:
    candidates: list[tuple[float, float, float]] = []
    for collaborative in [0.30, 0.45, 0.55, 0.65, 0.75]:
        for content in [0.15, 0.25, 0.35, 0.45, 0.55]:
            for popularity in [0.0, 0.05, 0.10, 0.15, 0.20]:
                total = collaborative + content + popularity
                if total <= 0:
                    continue
                candidates.append((collaborative / total, content / total, popularity / total))
    return sorted(set(candidates))


def set_weights(model: HybridMovieRecommender, weights: tuple[float, float, float]) -> None:
    model.alpha, model.beta, model.popularity_weight = weights


def relevant_by_user(frame: pd.DataFrame, positive_threshold: float) -> dict[int, set[int]]:
    positives = frame.loc[frame["rating"] >= positive_threshold]
    grouped = positives.groupby("userId")["movieId"].apply(lambda values: set(int(value) for value in values))
    return {int(user_id): values for user_id, values in grouped.items()}


def limit_holdout_users(frame: pd.DataFrame, positive_threshold: float, limit: int, seed: int) -> pd.DataFrame:
    if limit <= 0:
        return frame
    relevant = relevant_by_user(frame, positive_threshold)
    if len(relevant) <= limit:
        return frame
    selected = set(random.Random(seed).sample(sorted(relevant), limit))
    return frame.loc[frame["userId"].astype(int).isin(selected)].copy()


def build_component_cache(
    model: HybridMovieRecommender,
    user_ids: set[int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]]:
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]] = {}
    popularity = model._scale_scores(model._popularity_scores())
    for user_id in sorted(user_ids):
        collaborative = model._scale_scores(model._collaborative_scores(user_id))
        content = model._scale_scores(model._content_scores(user_id, []))
        excluded = [model.movie_index[movie_id] for movie_id in model.seen_movies(user_id) if movie_id in model.movie_index]
        cache[user_id] = (collaborative, content, popularity, excluded)
    return cache


def rank_from_components(
    model: HybridMovieRecommender,
    components: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    weights: tuple[float, float, float],
    top_k: int,
) -> list[int]:
    collaborative, content, popularity, excluded = components
    scores = weights[0] * collaborative + weights[1] * content + weights[2] * popularity
    if excluded:
        scores = scores.copy()
        scores[excluded] = -np.inf
    ranked = np.argsort(scores)[::-1]
    return [model.movie_ids[int(idx)] for idx in ranked[:top_k] if np.isfinite(scores[int(idx)])]


def evaluate_top_k(
    model: HybridMovieRecommender,
    holdout: pd.DataFrame,
    top_k: int,
    positive_threshold: float,
    weights: tuple[float, float, float],
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]],
) -> dict[str, float]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    mrr_values: list[float] = []

    for user_id, relevant in relevant_by_user(holdout, positive_threshold).items():
        components = cache.get(user_id)
        if components is None:
            continue
        recommended = rank_from_components(model, components, weights, top_k)
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


def evaluate_rating_rmse(model: HybridMovieRecommender, holdout: pd.DataFrame) -> float:
    y_true = []
    y_pred = []
    for row in holdout.itertuples():
        y_true.append(float(row.rating))
        y_pred.append(model.predict_rating(int(row.userId), int(row.movieId)))
    return rmse(y_true, y_pred)


def build_model(args: argparse.Namespace, bundle: Any, train: pd.DataFrame) -> HybridMovieRecommender:
    model = HybridMovieRecommender(
        min_rating=args.positive_threshold,
        content_backend=args.content_backend,
        content_model_name=getattr(args, "sbert_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    artifact_dir = Path(args.cf_artifact_dir) if args.cf_artifact_dir else Path(args.artifact_root) / f"recbole-{args.cf_model.lower()}"
    if artifact_dir.exists() and (artifact_dir / "manifest.json").exists():
        try:
            return model.load_artifact(artifact_dir, bundle.movies, train, bundle.tags)
        except Exception as exc:
            print(f"warning: failed to load collaborative artifact {artifact_dir}: {exc}", file=sys.stderr)
    return model.fit(bundle.movies, train, bundle.tags)


def tune(args: argparse.Namespace) -> dict[str, Any]:
    loader = MovieLensDataLoader(args.data_dir)
    bundle = loader.load()
    train, val, test = loader.train_val_test_split(bundle.ratings)
    eval_user_limit = int(getattr(args, "eval_user_limit", 0))
    seed = int(getattr(args, "seed", 42))
    val = limit_holdout_users(val, args.positive_threshold, eval_user_limit, seed)
    test = limit_holdout_users(test, args.positive_threshold, eval_user_limit, seed + 1)
    warm_val, cold_val = loader.split_warm_cold_items(train, val)
    warm_test, cold_test = loader.split_warm_cold_items(train, test)
    model = build_model(args, bundle, train)
    validation_relevant = relevant_by_user(val, args.positive_threshold)
    test_relevant = relevant_by_user(test, args.positive_threshold)
    component_cache = build_component_cache(model, set(validation_relevant) | set(test_relevant))

    best_weights = (model.alpha, model.beta, model.popularity_weight)
    best_metrics: dict[str, float] = {}
    best_key = (-1.0, -1.0, -1.0, -1.0)

    for weights in candidate_weights():
        metrics = evaluate_top_k(model, val, args.top_k, args.positive_threshold, weights, component_cache)
        key = (
            metrics[f"ndcg@{args.top_k}"],
            metrics[f"mrr@{args.top_k}"],
            metrics[f"recall@{args.top_k}"],
            metrics[f"precision@{args.top_k}"],
        )
        if key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics

    set_weights(model, best_weights)
    test_metrics = evaluate_top_k(model, test, args.top_k, args.positive_threshold, best_weights, component_cache)
    warm_test_metrics = evaluate_top_k(model, warm_test, args.top_k, args.positive_threshold, best_weights, component_cache)
    cold_test_metrics = evaluate_top_k(model, cold_test, args.top_k, args.positive_threshold, best_weights, component_cache)
    test_metrics["rmse"] = evaluate_rating_rmse(model, test)
    dataset_name = args.dataset_name or dataset_name_from_dir(args.data_dir)
    model_info = model.model_info()
    content_backend_used = str(model_info.get("content_backend", args.content_backend))
    result = {
        "dataset": dataset_name,
        "cf_model": args.cf_model,
        "content_backend": content_backend_used,
        "weights": {
            "collaborative": best_weights[0],
            "content": best_weights[1],
            "popularity": best_weights[2],
        },
        "validation": best_metrics,
        "validation_segments": {
            "warm_interactions": len(warm_val),
            "cold_interactions": len(cold_val),
        },
        "test": test_metrics,
        "test_segments": {
            "warm_interactions": len(warm_test),
            "cold_interactions": len(cold_test),
            "warm": warm_test_metrics,
            "cold": cold_test_metrics,
        },
        "model_info": model_info,
    }

    output_dir = Path(args.output_dir)
    manifest = model.save_artifact(
        output_dir,
        dataset_name=dataset_name,
        model_name=f"hybrid-{args.cf_model.lower()}-{content_backend_used}",
        metrics={
            "validation": best_metrics,
            "test": test_metrics,
            "test_segments": {
                "warm_interactions": len(warm_test),
                "cold_interactions": len(cold_test),
                "warm": warm_test_metrics,
                "cold": cold_test_metrics,
            },
        },
        extra_manifest={
            "tuning": {
                "primary_metric": f"ndcg@{args.top_k}",
                "tie_breakers": [f"mrr@{args.top_k}", f"recall@{args.top_k}", f"precision@{args.top_k}"],
                "candidate_count": len(candidate_weights()),
            }
        },
    )
    result["artifact_dir"] = str(output_dir)
    result["manifest"] = manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tuning_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hybrid recommender weights and export an artifact-first model.")
    parser.add_argument("--data-dir", default="data/ml-latest-small")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--cf-model", default="LightGCN")
    parser.add_argument("--cf-artifact-dir", default="")
    parser.add_argument("--artifact-root", default="artifacts/recommender")
    parser.add_argument("--content-backend", choices=["tfidf", "sbert", "auto"], default="tfidf")
    parser.add_argument("--sbert-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-dir", default="artifacts/recommender/latest")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--eval-user-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    result = tune(parse_args())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
