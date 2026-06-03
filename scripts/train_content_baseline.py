from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from models import HybridMovieRecommender
from scripts.train_baseline import evaluate_top_k_segments, prefixed


def evaluate(data_dir: str, top_k: int, artifact_dir: str = "", dataset_name: str = "") -> dict[str, float]:
    loader = MovieLensDataLoader(data_dir)
    bundle = loader.load()
    train, _, test = loader.train_val_test_split(bundle.ratings)
    warm_test, cold_test = loader.split_warm_cold_items(train, test)
    model = HybridMovieRecommender(
        alpha=0.0,
        beta=1.0,
        popularity_weight=0.0,
        collaborative_epochs=0,
    ).fit(bundle.movies, train, bundle.tags)
    model.model_name = "content-tfidf"
    model.model_source = "content_baseline"

    segment_metrics = evaluate_top_k_segments(
        model,
        {"all": test, "warm": warm_test, "cold": cold_test},
        top_k,
    )
    metrics = {
        **segment_metrics["all"],
        **prefixed("warm", segment_metrics["warm"]),
        **prefixed("cold", segment_metrics["cold"]),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a pure TF-IDF content-profile recommender.")
    parser.add_argument("--data-dir", default="data/sample")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--dataset-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.data_dir, args.top_k, args.artifact_dir, args.dataset_name)
    for name, value in metrics.items():
        print(f"{name}: {float(value):.4f}")


if __name__ == "__main__":
    main()
