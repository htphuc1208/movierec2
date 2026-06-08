#!/usr/bin/env python3
"""Create a 2D visualization of exported movie content embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize movie content embeddings from artifacts")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "embedding_visualization")
    parser.add_argument("--sample-size", type=int, default=2500)
    parser.add_argument("--top-genres", type=int, default=8)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = args.artifacts_dir / "movie_catalog.parquet"
    embedding_path = args.artifacts_dir / "content_embeddings.npy"
    if not catalog_path.exists() or not embedding_path.exists():
        raise FileNotFoundError(f"Missing {catalog_path} or {embedding_path}")

    catalog = pd.read_parquet(catalog_path).reset_index(drop=True)
    embeddings = np.load(embedding_path).astype(np.float32)
    if len(catalog) != embeddings.shape[0]:
        raise ValueError(f"catalog rows={len(catalog)} but embeddings rows={embeddings.shape[0]}")

    catalog = catalog.copy()
    catalog["primary_genre"] = _primary_genres(catalog)
    catalog = catalog.loc[~catalog["primary_genre"].isin(["", "Unknown", "(no genres listed)"])].copy()
    top_genres = catalog["primary_genre"].value_counts().head(args.top_genres).index.tolist()
    candidates = catalog.index[catalog["primary_genre"].isin(top_genres)].to_numpy(dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("No movies with usable genres were found")

    rng = np.random.default_rng(args.seed)
    sample_size = min(args.sample_size, candidates.size)
    sample_indices = rng.choice(candidates, size=sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]

    coords = _reduce(sample_embeddings, method=args.method, perplexity=args.perplexity, seed=args.seed)
    plot_df = catalog.loc[sample_indices, ["movieId", "title", "primary_genre"]].copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "embedding_visualization_points.csv"
    html_path = args.output_dir / "embedding_visualization.html"
    plot_df.to_csv(csv_path, index=False)
    _write_html(plot_df, html_path, method=args.method)
    print(f"csv: {csv_path}")
    print(f"html: {html_path}")


def _primary_genres(catalog: pd.DataFrame) -> pd.Series:
    genre_col = "tmdb_genres" if "tmdb_genres" in catalog.columns else "genres"
    if genre_col not in catalog.columns:
        return pd.Series(["Unknown"] * len(catalog), index=catalog.index)
    return (
        catalog[genre_col]
        .fillna("")
        .astype(str)
        .str.replace(",", "|")
        .str.split("|")
        .str[0]
        .fillna("Unknown")
        .str.strip()
    )


def _reduce(embeddings: np.ndarray, method: str, perplexity: float, seed: int) -> np.ndarray:
    if embeddings.shape[0] < 3:
        coords = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
        if embeddings.shape[0] == 2:
            coords[:, 0] = np.asarray([0.0, 1.0], dtype=np.float32)
        return coords
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    safe_perplexity = min(float(perplexity), max(2.0, float(embeddings.shape[0] - 1)))
    return TSNE(n_components=2, random_state=seed, perplexity=safe_perplexity, init="pca", learning_rate="auto").fit_transform(embeddings)


def _write_html(plot_df: pd.DataFrame, output_path: Path, method: str) -> None:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("visualize_embeddings.py requires plotly. Install requirements.txt first.") from exc
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="primary_genre",
        hover_data=["movieId", "title"],
        title=f"{method.upper()} visualization of movie content embeddings",
    )
    fig.update_traces(marker={"size": 7, "opacity": 0.8})
    fig.write_html(output_path)


if __name__ == "__main__":
    main()
