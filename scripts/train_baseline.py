from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse
from models import HybridMovieRecommender


def evaluate_top_k_segments(
    model: HybridMovieRecommender,
    segments: dict[str, object],
    top_k: int,
) -> dict[str, dict[str, float]]:
    relevant_segments = {name: relevant_sets(model, holdout) for name, holdout in segments.items()}
    user_ids = sorted({user_id for relevant in relevant_segments.values() for user_id in relevant})
    recommendations_by_user = {
        user_id: rank_movie_ids(model, user_id, top_k)
        for user_id in user_ids
    }
    return {
        name: metrics_from_rankings(recommendations_by_user, relevant, top_k)
        for name, relevant in relevant_segments.items()
    }


def rank_movie_ids(model: HybridMovieRecommender, user_id: int, top_k: int) -> list[int]:
    cf_scores = model._collaborative_scores(user_id)
    content_scores = model._content_scores(user_id, [])
    popularity_scores = model._popularity_scores()
    scores = (
        model.alpha * model._scale_scores(cf_scores)
        + model.beta * model._scale_scores(content_scores)
        + model.popularity_weight * model._scale_scores(popularity_scores)
    )
    for movie_id in model.seen_movies(user_id):
        idx = model.movie_index.get(movie_id)
        if idx is not None:
            scores[idx] = -np.inf
    ranked = np.argsort(scores)[::-1]
    return [model.movie_ids[int(idx)] for idx in ranked[:top_k] if np.isfinite(scores[int(idx)])]


def relevant_sets(model: HybridMovieRecommender, holdout) -> dict[int, set[int]]:
    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    for row in holdout.itertuples():
        if float(row.rating) >= model.min_rating:
            relevant_by_user[int(row.userId)].add(int(row.movieId))
    return relevant_by_user


def metrics_from_rankings(
    recommendations_by_user: dict[int, list[int]],
    relevant_by_user: dict[int, set[int]],
    top_k: int,
) -> dict[str, float]:
    precision_values = []
    recall_values = []
    ndcg_values = []
    mrr_values = []
    for user_id, relevant in relevant_by_user.items():
        recommended_ids = recommendations_by_user.get(user_id, [])
        precision_values.append(precision_at_k(recommended_ids, relevant, top_k))
        recall_values.append(recall_at_k(recommended_ids, relevant, top_k))
        ndcg_values.append(ndcg_at_k(recommended_ids, relevant, top_k))
        mrr_values.append(mrr_at_k(recommended_ids, relevant, top_k))

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        f"precision@{top_k}": avg(precision_values),
        f"recall@{top_k}": avg(recall_values),
        f"ndcg@{top_k}": avg(ndcg_values),
        f"mrr@{top_k}": avg(mrr_values),
    }


def evaluate_rating_segment(model: HybridMovieRecommender, holdout) -> float:
    y_true = []
    y_pred = []
    for row in holdout.itertuples():
        y_true.append(float(row.rating))
        y_pred.append(model.predict_rating(int(row.userId), int(row.movieId)))
    return rmse(y_true, y_pred)


def prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def evaluate(data_dir: str, top_k: int, artifact_dir: str = "", dataset_name: str = "") -> dict[str, float]:
    loader = MovieLensDataLoader(data_dir)
    bundle = loader.load()
    train, _, test = loader.train_val_test_split(bundle.ratings)
    warm_test, cold_test = loader.split_warm_cold_items(train, test)
    model = HybridMovieRecommender().fit(bundle.movies, train, bundle.tags)

    segment_metrics = evaluate_top_k_segments(
        model,
        {"all": test, "warm": warm_test, "cold": cold_test},
        top_k,
    )
    all_metrics = segment_metrics["all"]
    warm_metrics = segment_metrics["warm"]
    cold_metrics = segment_metrics["cold"]
    rating_rmse = evaluate_rating_segment(model, test)
    metrics = {
        **all_metrics,
        **prefixed("warm", warm_metrics),
        **prefixed("cold", cold_metrics),
        "rating_rmse": rating_rmse,
        "rmse": rating_rmse,
        "warm_test_interactions": float(len(warm_test)),
        "cold_test_interactions": float(len(cold_test)),
    }
    if artifact_dir:
        model.save_artifact(
            artifact_dir,
            dataset_name=dataset_name or Path(data_dir).resolve().name,
            model_name=model.model_name,
            metrics={"test": metrics},
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the lightweight hybrid baseline.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--dataset-name", default="")
    args = parser.parse_args()

    metrics = evaluate(args.data_dir, args.top_k, args.artifact_dir, args.dataset_name)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
