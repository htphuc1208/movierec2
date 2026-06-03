from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from models import HybridMovieRecommender


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


@lru_cache(maxsize=1)
def local_recommender() -> HybridMovieRecommender:
    data_dir = os.getenv("MOVIEREC_DATA_DIR", "data/sample")
    bundle = MovieLensDataLoader(data_dir).load()
    model = HybridMovieRecommender()
    artifact_dir = os.getenv("MOVIEREC_ARTIFACT_DIR", "").strip()
    if artifact_dir:
        try:
            model.load_artifact(artifact_dir, bundle.movies, bundle.ratings, bundle.tags)
            if not model.dataset_name:
                model.dataset_name = data_dir
            return model
        except Exception:
            pass
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    model.dataset_name = data_dir
    return model


def api_get(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=1.5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def load_movies() -> list[dict[str, Any]]:
    data = api_get("/movies")
    if data and "movies" in data:
        return data["movies"]
    return local_recommender().movies_for_picker()


def load_users() -> list[int]:
    data = api_get("/users")
    if data and "users" in data:
        return data["users"]
    return local_recommender().users()


def load_model_info() -> dict[str, Any]:
    data = api_get("/model-info")
    if data and "model_info" in data:
        return data["model_info"]
    return local_recommender().model_info()


def recommend(user_id: int | None, top_k: int, session_context: list[str], exclude_seen: bool) -> list[dict[str, Any]]:
    payload = {
        "user_id": user_id,
        "top_k": top_k,
        "session_context": session_context,
        "exclude_seen": exclude_seen,
    }
    data = api_post("/recommend", payload)
    if data and "recommendations" in data:
        return data["recommendations"]
    return local_recommender().recommend(
        user_id=user_id,
        top_k=top_k,
        session_context=session_context,
        exclude_seen=exclude_seen,
    )


def load_movie_collection(path: str) -> list[dict[str, Any]]:
    data = api_get(path)
    if data and "movies" in data:
        return data["movies"]
    return []


def load_user_history(user_id: int, top_k: int = 15) -> list[dict[str, Any]]:
    return load_movie_collection(f"/users/{user_id}/history?top_k={top_k}")


def render_card(movie: dict[str, Any]) -> None:
    poster_url = movie.get("poster_url") or ""
    title = movie.get("title", "Untitled")
    score = float(movie.get("score") or movie.get("rating_mean") or 0.0)
    vote_count = int(movie.get("vote_count") or 0)
    genres = movie.get("genres", "")
    overview = movie.get("overview", "")
    reasons = movie.get("reason", [])
    cf_score = float(movie.get("collaborative_score", 0.0))
    content_score = float(movie.get("content_score", 0.0))
    popularity_score = float(movie.get("popularity_score", 0.0))

    with st.container(border=True):
        left, right = st.columns([1, 2], vertical_alignment="top")
        with left:
            if poster_url:
                st.image(poster_url, width="stretch")
            else:
                st.markdown(f"**{title}**")
        with right:
            st.markdown(f"### {title}")
            st.caption(f"Score {score:.3f} | Votes {vote_count} | {genres}")
            if cf_score or content_score or popularity_score:
                st.caption(f"CF {cf_score:.3f} | Content {content_score:.3f} | Popularity {popularity_score:.3f}")
            if "user_rating" in movie:
                st.caption(f"Your rating {float(movie['user_rating']):.1f}")
            if overview:
                st.write(overview)
            if reasons:
                st.write(" | ".join(str(reason) for reason in reasons))


def render_movie_grid(movies: list[dict[str, Any]], columns: int = 3) -> None:
    if not movies:
        st.info("No movies to show.")
        return
    for row_start in range(0, len(movies), columns):
        row_columns = st.columns(columns)
        for column, movie in zip(row_columns, movies[row_start : row_start + columns]):
            with column:
                render_card(movie)


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="film", layout="wide")
    st.title("Hybrid Movie Recommender")

    movies = load_movies()
    users = load_users()
    model_info = load_model_info()
    movie_frame = pd.DataFrame(movies)

    with st.sidebar:
        st.caption(f"{model_info.get('model_name', 'hybrid')} | {model_info.get('model_source', 'runtime')}")
        metrics = model_info.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            flat_metrics = metrics.get("test") if isinstance(metrics.get("test"), dict) else metrics
            if isinstance(flat_metrics, dict):
                ndcg = flat_metrics.get("ndcg@10") or flat_metrics.get("NDCG@10")
                mrr = flat_metrics.get("mrr@10") or flat_metrics.get("MRR@10")
                if ndcg is not None or mrr is not None:
                    st.caption(f"NDCG@10 {float(ndcg or 0):.4f} | MRR@10 {float(mrr or 0):.4f}")
        user_options = ["Guest"] + [str(user) for user in users]
        selected_user = st.selectbox("User", user_options, index=0 if not users else 1)
        top_k = st.slider("Top K", 5, 20, 10, 1)
        exclude_seen = st.toggle("Hide watched", value=True)

    selected_titles = st.multiselect(
        "Session movies",
        options=movie_frame["title"].tolist(),
        default=[],
        max_selections=5,
    )
    selected_ids = movie_frame.loc[movie_frame["title"].isin(selected_titles), "movieId"].astype(str).tolist()
    user_id = None if selected_user == "Guest" else int(selected_user)

    recommend_tab, discover_tab, history_tab = st.tabs(["Recommendations", "Discover", "History"])

    with recommend_tab:
        recommendations = recommend(user_id, top_k, selected_ids, exclude_seen)
        metric_cols = st.columns(3)
        metric_cols[0].metric("Results", len(recommendations))
        metric_cols[1].metric("User", "Guest" if user_id is None else user_id)
        metric_cols[2].metric("Session", len(selected_ids))
        render_movie_grid(recommendations, columns=2)

    with discover_tab:
        search = st.text_input("Search movies")
        if search.strip():
            render_movie_grid(load_movie_collection(f"/movies?search={quote_plus(search.strip())}")[:top_k], columns=2)
        else:
            collections = {
                "Trending": "/movies/trending?top_k=12",
                "Top Rated": "/movies/top-rated?top_k=12",
                "Latest": "/movies/latest?top_k=12",
                "Action": "/movies/genre/Action?top_k=12",
                "Comedy": "/movies/genre/Comedy?top_k=12",
            }
            selected_collection = st.selectbox("Collection", list(collections))
            render_movie_grid(load_movie_collection(collections[selected_collection]), columns=3)

    with history_tab:
        if user_id is None:
            st.info("Select a user to view rating history.")
        else:
            render_movie_grid(load_user_history(user_id, top_k=top_k), columns=2)


if __name__ == "__main__":
    main()
