"""Streamlit EDA dashboard for MovieLens and Letterboxd processed data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from recommender.config import PROJECT_ROOT
from recommender.analysis.eda import (
    genre_counts,
    infer_interaction_columns,
    load_catalog,
    load_interactions,
    overview_stats,
    rating_distribution,
    top_interacted_movies,
    user_segmentation,
)


DATASETS = {
    "MovieLens": {
        "ratings": PROJECT_ROOT / "data" / "raw" / "ml-latest-small" / "ratings.csv",
        "catalog": PROJECT_ROOT / "data" / "processed" / "movie_catalog_enriched.parquet",
    },
    "Letterboxd": {
        "ratings": PROJECT_ROOT / "data" / "processed" / "letterboxd" / "ratings.csv",
        "catalog": PROJECT_ROOT / "data" / "processed" / "letterboxd" / "movie_catalog_enriched.parquet",
    },
}


@st.cache_data
def cached_interactions(path: str) -> pd.DataFrame:
    return load_interactions(path)


@st.cache_data
def cached_catalog(path: str) -> pd.DataFrame:
    return load_catalog(path)


def main() -> None:
    st.set_page_config(page_title="movierec EDA", layout="wide")
    st.title("Phân tích dữ liệu gợi ý phim")

    source = st.sidebar.selectbox("Nguồn dữ liệu", list(DATASETS))
    paths = DATASETS[source]
    ratings_path = Path(paths["ratings"])
    catalog_path = Path(paths["catalog"])
    st.sidebar.caption(f"Ratings: {ratings_path}")
    st.sidebar.caption(f"Catalog: {catalog_path}")

    if not ratings_path.exists() or not catalog_path.exists():
        st.warning("Thiếu ratings hoặc catalog enriched. Hãy chạy prepare/enrich trước.")
        return

    interactions = cached_interactions(str(ratings_path))
    catalog = cached_catalog(str(catalog_path))
    columns = infer_interaction_columns(interactions)
    stats = overview_stats(interactions, columns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users", f"{stats['users']:,}")
    col2.metric("Phim có tương tác", f"{stats['items']:,}")
    col3.metric("Interactions", f"{stats['interactions']:,}")
    col4.metric("Sparsity", f"{stats['sparsity'] * 100:.2f}%")

    tab_dist, tab_meta, tab_users, tab_top = st.tabs(["Phân phối", "Metadata", "User Segmentation", "Top-K"])

    with tab_dist:
        ratings = rating_distribution(interactions, columns)
        st.plotly_chart(
            px.bar(ratings, x="rating", y="count", title=f"Phân phối rating - {source}", text_auto=True),
            use_container_width=True,
        )

    with tab_meta:
        left, right = st.columns(2)
        with left:
            if "release_year" in catalog.columns:
                st.plotly_chart(px.histogram(catalog, x="release_year", nbins=50, title="Năm phát hành"), use_container_width=True)
            runtime_col = "runtime_minutes" if "runtime_minutes" in catalog.columns else "runtime"
            if runtime_col in catalog.columns:
                runtime = catalog[pd.to_numeric(catalog[runtime_col], errors="coerce") > 0]
                st.plotly_chart(px.histogram(runtime, x=runtime_col, nbins=50, title="Thời lượng phim"), use_container_width=True)
        with right:
            if "popularity" in catalog.columns:
                st.plotly_chart(px.histogram(catalog, x="popularity", nbins=50, log_y=True, title="TMDb popularity"), use_container_width=True)
            if "vote_average" in catalog.columns:
                votes = catalog[pd.to_numeric(catalog["vote_average"], errors="coerce") > 0]
                st.plotly_chart(px.histogram(votes, x="vote_average", nbins=20, title="TMDb vote average"), use_container_width=True)

    with tab_users:
        method_label = st.selectbox("Thuật toán", ["KMeans", "GMM"])
        n_clusters = st.slider("Số cụm", min_value=2, max_value=8, value=4)
        method = "gmm" if method_label == "GMM" else "kmeans"
        with st.spinner("Đang phân cụm user..."):
            clusters, genre_cols = user_segmentation(interactions, catalog, columns, n_clusters=n_clusters, method=method)
        if clusters.empty:
            st.info("Không đủ feature để phân cụm user.")
        else:
            st.plotly_chart(
                px.scatter(
                    clusters,
                    x="PCA1",
                    y="PCA2",
                    color="cluster_name",
                    hover_data=[columns.user],
                    title=f"User segmentation bằng {method_label}",
                ),
                use_container_width=True,
            )
            profile = clusters.groupby("cluster_name").mean(numeric_only=True).reset_index()
            show_cols = ["cluster_name"] + [col for col in ["vote_average", "popularity", "release_year"] if col in profile.columns] + genre_cols[:6]
            st.dataframe(profile[show_cols], use_container_width=True, hide_index=True)

    with tab_top:
        left, right = st.columns(2)
        with left:
            genres = genre_counts(catalog, top_n=20)
            if not genres.empty:
                st.plotly_chart(px.bar(genres, x="count", y="genre", orientation="h", title="Top thể loại"), use_container_width=True)
        with right:
            top_movies = top_interacted_movies(interactions, catalog, columns, top_n=20)
            st.plotly_chart(px.bar(top_movies, x="interactions", y="title", orientation="h", title="Top phim theo tương tác"), use_container_width=True)


if __name__ == "__main__":
    main()
