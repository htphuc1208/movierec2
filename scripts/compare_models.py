#!/usr/bin/env python3
"""Run offline comparison across recommender models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config import PROJECT_ROOT
from recommender.experiments.comparison import ComparisonConfig, run_comparison, write_comparison_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recommender models on MovieLens and/or Letterboxd splits")
    # dataset options
    parser.add_argument("--dataset", choices=["movielens", "letterboxd", "both"], default="both")
    parser.add_argument("--movielens-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ml-latest-small")
    parser.add_argument(
        "--movielens-enriched-catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "movie_catalog_enriched.parquet",
    )
    parser.add_argument("--letterboxd-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "letterboxd")
    parser.add_argument(
        "--letterboxd-enriched-catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "letterboxd" / "movie_catalog_enriched.parquet",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "comparison")
    parser.add_argument("--content-backend", choices=["tfidf", "sbert", "auto"], default="tfidf")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    # core: chỉ chạy các model cơ bản như ItemPop, ItemKNN, SVD, MF
    # full: chạy tất cả model bao gồm cả EASE, SLIM, Ranker
    parser.add_argument("--models", choices=["core", "full"], default="core")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--knn-top-k", type=int, default=100)
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument("--mf-dim", type=int, default=64)
    # giới hạn số lượng item được sử dụng để huấn luyện EASE và SLIM để giảm thời gian chạy, đồng thời giới hạn số lượng mẫu được sử dụng để huấn luyện Ranker
    parser.add_argument("--max-ease-items", type=int, default=8000)
    parser.add_argument("--max-slim-items", type=int, default=1000)
    parser.add_argument("--max-ranker-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[tuple[str, Path, Path | None]]:
    datasets: list[tuple[str, Path, Path | None]] = []
    if args.dataset in {"movielens", "both"}:
        datasets.append(("movielens", args.movielens_dir, args.movielens_enriched_catalog))
    if args.dataset in {"letterboxd", "both"}:
        datasets.append(("letterboxd", args.letterboxd_dir, args.letterboxd_enriched_catalog))
    return datasets


def main() -> None:
    args = parse_args()
    config = ComparisonConfig(
        k=args.k,
        batch_size=args.batch_size,
        models=args.models,
        content_backend=args.content_backend,
        sbert_model=args.sbert_model,
        min_rating=args.min_rating,
        epochs=args.epochs,
        device=args.device,
        knn_top_k=args.knn_top_k,
        svd_components=args.svd_components,
        mf_dim=args.mf_dim,
        max_ease_items=args.max_ease_items,
        max_slim_items=args.max_slim_items,
        max_ranker_samples=args.max_ranker_samples,
        seed=args.seed,
    )
    rows = run_comparison(selected_datasets(args), config)
    paths = write_comparison_outputs(rows, args.output_dir, k=args.k)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
