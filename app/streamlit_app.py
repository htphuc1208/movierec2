from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

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
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
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


def render_card(movie: dict[str, Any]) -> None:
    poster_url = movie.get("poster_url") or ""
    title = movie.get("title", "Untitled")
    score = float(movie.get("score", 0.0))
    genres = movie.get("genres", "")
    overview = movie.get("overview", "")
    reasons = movie.get("reason", [])

    with st.container(border=True):
        left, right = st.columns([1, 2], vertical_alignment="top")
        with left:
            if poster_url:
                st.image(poster_url, width="stretch")
            else:
                st.markdown(f"**{title}**")
        with right:
            st.markdown(f"### {title}")
            st.caption(f"Score {score:.3f} | {genres}")
            if overview:
                st.write(overview)
            if reasons:
                st.write(" | ".join(str(reason) for reason in reasons))


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="film", layout="wide")
    st.title("Hybrid Movie Recommender")

    movies = load_movies()
    users = load_users()
    movie_frame = pd.DataFrame(movies)

    with st.sidebar:
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

    recommendations = recommend(user_id, top_k, selected_ids, exclude_seen)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Results", len(recommendations))
    metric_cols[1].metric("User", "Guest" if user_id is None else user_id)
    metric_cols[2].metric("Session", len(selected_ids))

    for row_start in range(0, len(recommendations), 2):
        columns = st.columns(2)
        for column, movie in zip(columns, recommendations[row_start : row_start + 2]):
            with column:
                render_card(movie)


if __name__ == "__main__":
    main()
