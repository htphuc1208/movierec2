"""Reusable EDA helpers for movie recommendation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


ClusterMethod = Literal["kmeans", "gmm"]


@dataclass(frozen=True)
class InteractionColumns:
    user: str
    item: str
    rating: str


def load_interactions(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_catalog(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def infer_interaction_columns(interactions: pd.DataFrame) -> InteractionColumns:
    lower_to_original = {column.lower(): column for column in interactions.columns}
    user = lower_to_original.get("userid") or lower_to_original.get("user_id") or interactions.columns[0]
    item = lower_to_original.get("movieid") or lower_to_original.get("movie_id") or interactions.columns[1]
    rating = lower_to_original.get("rating") or interactions.columns[2]
    return InteractionColumns(user=str(user), item=str(item), rating=str(rating))


def overview_stats(interactions: pd.DataFrame, columns: InteractionColumns | None = None) -> dict[str, float]:
    columns = columns or infer_interaction_columns(interactions)
    num_users = int(interactions[columns.user].nunique())
    num_items = int(interactions[columns.item].nunique())
    num_interactions = int(len(interactions))
    total_cells = max(1, num_users * num_items)
    sparsity = 1.0 - (num_interactions / total_cells)
    return {
        "users": num_users,
        "items": num_items,
        "interactions": num_interactions,
        "sparsity": float(max(0.0, min(1.0, sparsity))),
    }


def rating_distribution(interactions: pd.DataFrame, columns: InteractionColumns | None = None) -> pd.DataFrame:
    columns = columns or infer_interaction_columns(interactions)
    values = interactions[columns.rating].value_counts().sort_index().reset_index()
    values.columns = ["rating", "count"]
    return values


def genre_counts(catalog: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    genre_col = "tmdb_genres" if "tmdb_genres" in catalog.columns else "genres"
    if genre_col not in catalog.columns:
        return pd.DataFrame(columns=["genre", "count"])
    values = catalog[genre_col].fillna("").astype(str).str.replace(",", "|").str.split("|").explode().str.strip()
    values = values[(values != "") & (values != "(no genres listed)")]
    counts = values.value_counts().head(top_n).reset_index()
    counts.columns = ["genre", "count"]
    return counts


def top_interacted_movies(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    columns: InteractionColumns | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    columns = columns or infer_interaction_columns(interactions)
    counts = interactions[columns.item].value_counts().head(top_n).reset_index()
    counts.columns = ["movieId", "interactions"]
    catalog_key = "movieId" if "movieId" in catalog.columns else catalog.columns[0]
    clean_counts = counts.copy()
    clean_catalog = catalog[[catalog_key, "title"]].copy() if "title" in catalog.columns else catalog[[catalog_key]].copy()
    clean_counts["movieId"] = pd.to_numeric(clean_counts["movieId"], errors="coerce").astype("Int64")
    clean_catalog[catalog_key] = pd.to_numeric(clean_catalog[catalog_key], errors="coerce").astype("Int64")
    merged = clean_counts.merge(clean_catalog, left_on="movieId", right_on=catalog_key, how="left")
    if "title" not in merged.columns:
        merged["title"] = merged["movieId"].astype(str)
    return merged[["movieId", "title", "interactions"]]


def user_segmentation(
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
    columns: InteractionColumns | None = None,
    n_clusters: int = 4,
    method: ClusterMethod = "kmeans",
) -> tuple[pd.DataFrame, list[str]]:
    columns = columns or infer_interaction_columns(interactions)
    catalog_key = "movieId" if "movieId" in catalog.columns else catalog.columns[0]
    inter = interactions[[columns.user, columns.item]].copy()
    meta = catalog.copy()
    inter[columns.item] = pd.to_numeric(inter[columns.item], errors="coerce").astype("Int64")
    meta[catalog_key] = pd.to_numeric(meta[catalog_key], errors="coerce").astype("Int64")

    numeric_cols = [column for column in ["popularity", "vote_average", "release_year", "runtime_minutes"] if column in meta.columns]
    for column in numeric_cols:
        meta[column] = pd.to_numeric(meta[column], errors="coerce").fillna(0.0)

    genre_col = "tmdb_genres" if "tmdb_genres" in meta.columns else "genres"
    genre_cols: list[str] = []
    if genre_col in meta.columns:
        genre_matrix = meta[genre_col].fillna("").astype(str).str.replace(",", "|").str.get_dummies(sep="|")
        genre_matrix = genre_matrix.loc[:, [column for column in genre_matrix.columns if column and column != "(no genres listed)"]]
        genre_cols = [str(column) for column in genre_matrix.columns]
        meta = pd.concat([meta, genre_matrix], axis=1)

    feature_cols = numeric_cols + genre_cols
    if not feature_cols:
        return pd.DataFrame(), []
    merged = inter.merge(meta[[catalog_key, *feature_cols]], left_on=columns.item, right_on=catalog_key, how="inner")
    if merged.empty:
        return pd.DataFrame(), genre_cols
    user_features = merged.groupby(columns.user)[feature_cols].mean().reset_index()
    x = user_features[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    if len(user_features) < 2:
        return pd.DataFrame(), genre_cols

    n_clusters = max(2, min(int(n_clusters), len(user_features)))
    x_scaled = StandardScaler().fit_transform(x)
    if method == "gmm":
        clusters = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(x_scaled)
    else:
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(x_scaled)
    user_features["cluster"] = clusters.astype(int)
    user_features["cluster_name"] = user_features["cluster"].map(_cluster_names(user_features, genre_cols))

    components = min(2, x_scaled.shape[1], len(user_features))
    if components == 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
        user_features["PCA1"] = coords[:, 0]
        user_features["PCA2"] = coords[:, 1]
    else:
        user_features["PCA1"] = np.arange(len(user_features), dtype=np.float32)
        user_features["PCA2"] = 0.0
    return user_features, genre_cols


def _cluster_names(user_features: pd.DataFrame, genre_cols: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    if genre_cols:
        means = user_features.groupby("cluster")[genre_cols].mean()
        for cluster in sorted(user_features["cluster"].unique()):
            top_genre = str(means.loc[cluster].idxmax()) if cluster in means.index else ""
            result[int(cluster)] = f"Cụm {int(cluster)}: fan {top_genre}" if top_genre else f"Cụm {int(cluster)}"
    else:
        for cluster in sorted(user_features["cluster"].unique()):
            result[int(cluster)] = f"Cụm {int(cluster)}"
    return result
