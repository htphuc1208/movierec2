from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from evaluation import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k, rmse
from models import HybridMovieRecommender


def evaluate(data_dir: str, top_k: int, artifact_dir: str = "", dataset_name: str = "") -> dict[str, float]:
    loader = MovieLensDataLoader(data_dir)
    bundle = loader.load()
    train, _, test = loader.train_val_test_split(bundle.ratings)
    model = HybridMovieRecommender().fit(bundle.movies, train, bundle.tags)

    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    for row in test.itertuples():
        if float(row.rating) >= model.min_rating:
            relevant_by_user[int(row.userId)].add(int(row.movieId))

    precision_values = []
    recall_values = []
    ndcg_values = []
    mrr_values = []
    y_true = []
    y_pred = []

    for user_id, relevant in relevant_by_user.items():
        recs = model.recommend(user_id=user_id, top_k=top_k, exclude_seen=True)
        recommended_ids = [int(rec["movie_id"]) for rec in recs]
        precision_values.append(precision_at_k(recommended_ids, relevant, top_k))
        recall_values.append(recall_at_k(recommended_ids, relevant, top_k))
        ndcg_values.append(ndcg_at_k(recommended_ids, relevant, top_k))
        mrr_values.append(mrr_at_k(recommended_ids, relevant, top_k))

    for row in test.itertuples():
        y_true.append(float(row.rating))
        y_pred.append(model.predict_rating(int(row.userId), int(row.movieId)))

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    rating_rmse = rmse(y_true, y_pred)
    metrics = {
        f"precision@{top_k}": avg(precision_values),
        f"recall@{top_k}": avg(recall_values),
        f"ndcg@{top_k}": avg(ndcg_values),
        f"mrr@{top_k}": avg(mrr_values),
        "rating_rmse": rating_rmse,
        "rmse": rating_rmse,
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
