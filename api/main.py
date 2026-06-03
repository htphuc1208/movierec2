from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import MovieLensDataLoader
from models import HybridMovieRecommender


class RecommendRequest(BaseModel):
    user_id: int | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    session_context: list[str] = Field(default_factory=list)
    exclude_seen: bool = True
    model_name: str | None = None


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_info: dict[str, Any]


app = FastAPI(
    title="Hybrid Movie Recommendation API",
    version="0.1.0",
    description="MovieLens-style hybrid recommender with collaborative and metadata signals.",
)


@lru_cache(maxsize=1)
def get_recommender() -> HybridMovieRecommender:
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
        except Exception as exc:
            model.artifact_manifest = {"artifact_load_error": str(exc), "artifact_path": artifact_dir}
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    model.dataset_name = data_dir
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users")
def users() -> dict[str, list[int]]:
    return {"users": get_recommender().users()}


@app.get("/movies")
def movies(search: str | None = None) -> dict[str, list[dict[str, Any]]]:
    records = _movies_with_rating_stats()
    if search:
        query = search.lower().strip()
        records = [record for record in records if query in str(record.get("title", "")).lower()]
    return {"movies": records}


@app.get("/movies/trending")
def trending_movies(top_k: int = 15, min_votes: int = 15) -> dict[str, list[dict[str, Any]]]:
    records = [movie for movie in _movies_with_rating_stats() if int(movie.get("vote_count", 0)) >= min_votes]
    records.sort(key=lambda movie: int(movie.get("vote_count", 0)), reverse=True)
    return {"movies": records[: max(1, min(top_k, 50))]}


@app.get("/movies/top-rated")
def top_rated_movies(top_k: int = 15, min_votes: int = 15) -> dict[str, list[dict[str, Any]]]:
    records = [movie for movie in _movies_with_rating_stats() if int(movie.get("vote_count", 0)) >= min_votes]
    records.sort(key=lambda movie: (float(movie.get("rating_mean", 0.0)), int(movie.get("vote_count", 0))), reverse=True)
    return {"movies": records[: max(1, min(top_k, 50))]}


@app.get("/movies/latest")
def latest_movies(top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    records = _movies_with_rating_stats()
    records.sort(key=lambda movie: (int(movie.get("year") or 0), int(movie.get("vote_count", 0))), reverse=True)
    return {"movies": records[: max(1, min(top_k, 50))]}


@app.get("/movies/genre/{genre}")
def genre_movies(genre: str, top_k: int = 15, min_votes: int = 0) -> dict[str, list[dict[str, Any]]]:
    genre_query = genre.lower().strip()
    records = [
        movie
        for movie in _movies_with_rating_stats()
        if genre_query in str(movie.get("genres", "")).lower()
        and int(movie.get("vote_count", 0)) >= min_votes
    ]
    records.sort(key=lambda movie: (float(movie.get("rating_mean", 0.0)), int(movie.get("vote_count", 0))), reverse=True)
    return {"movies": records[: max(1, min(top_k, 50))]}


@app.get("/movies/{movie_id}")
def movie_details(movie_id: int) -> dict[str, Any]:
    recommender = get_recommender()
    if recommender.movies is None:
        raise HTTPException(status_code=404, detail="Movie catalog is not loaded")
    row = recommender.movies.loc[recommender.movies["movieId"].astype(int) == int(movie_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    record = _json_record(row.iloc[0].to_dict())
    stats = _rating_stats_by_movie().get(int(movie_id), {})
    record.update(stats)
    return record


@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: int, top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    recommender = get_recommender()
    if int(movie_id) not in recommender.movie_index:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Reuse the runtime content space instead of loading SBERT in the API process.
    scores = recommender._content_scores(user_id=None, session_movie_ids=[int(movie_id)])
    target_idx = recommender.movie_index[int(movie_id)]
    scores[target_idx] = -np.inf
    movie_by_id = {int(movie["movieId"]): movie for movie in _movies_with_rating_stats()}
    ranked = np.argsort(scores)[::-1]
    records: list[dict[str, Any]] = []
    for idx in ranked:
        if len(records) >= max(1, min(top_k, 50)):
            break
        if not np.isfinite(scores[int(idx)]):
            continue
        candidate_id = int(recommender.movie_ids[int(idx)])
        candidate = dict(movie_by_id.get(candidate_id, {}))
        if candidate:
            candidate["similarity_score"] = float(scores[int(idx)])
            records.append(candidate)
    return {"movies": records}


@app.get("/users/{user_id}/history")
def user_history(user_id: int, top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    recommender = get_recommender()
    if recommender.ratings is None:
        return {"movies": []}
    history = recommender.ratings.loc[recommender.ratings["userId"].astype(int) == int(user_id)].copy()
    if history.empty:
        return {"movies": []}
    history = history.sort_values(["rating", "timestamp"], ascending=[False, False]).head(max(1, min(top_k, 100)))
    movies_by_id = {int(movie["movieId"]): movie for movie in _movies_with_rating_stats()}
    records: list[dict[str, Any]] = []
    for row in history.itertuples():
        movie = dict(movies_by_id.get(int(row.movieId), {}))
        if movie:
            movie["user_rating"] = float(row.rating)
            movie["rated_at"] = int(row.timestamp)
            records.append(movie)
    return {"movies": records}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    info = get_recommender().model_info()
    if get_recommender().artifact_manifest.get("artifact_load_error"):
        info["artifact_load_error"] = get_recommender().artifact_manifest["artifact_load_error"]
    return ModelInfoResponse(model_info=info)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    try:
        recommendations = get_recommender().recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            session_context=request.session_context,
            exclude_seen=request.exclude_seen,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RecommendResponse(recommendations=recommendations)


def _rating_stats_by_movie() -> dict[int, dict[str, Any]]:
    recommender = get_recommender()
    if recommender.ratings is None or recommender.ratings.empty:
        return {}
    grouped = recommender.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    return {
        int(movie_id): {
            "rating_mean": float(row["mean"]),
            "score": float(row["mean"]),
            "vote_count": int(row["count"]),
        }
        for movie_id, row in grouped.iterrows()
    }


def _movies_with_rating_stats() -> list[dict[str, Any]]:
    recommender = get_recommender()
    stats = _rating_stats_by_movie()
    records: list[dict[str, Any]] = []
    if recommender.movies is None:
        return records
    for row in recommender.movies.itertuples():
        movie_id = int(row.movieId)
        record = {
            "movieId": movie_id,
            "movie_id": movie_id,
            "title": str(_json_scalar(getattr(row, "title", ""))),
            "genres": str(_json_scalar(getattr(row, "genres", ""))),
            "year": _extract_year_value(getattr(row, "year", ""), getattr(row, "title", "")),
            "tmdbId": _json_scalar(getattr(row, "tmdbId", "")),
            "poster_url": str(_json_scalar(getattr(row, "poster_url", ""))),
            "overview": str(_json_scalar(getattr(row, "overview", ""))),
            "director": str(_json_scalar(getattr(row, "director", ""))),
            "cast": str(_json_scalar(getattr(row, "cast", ""))),
        }
        record.update(stats.get(movie_id, {"rating_mean": 0.0, "score": 0.0, "vote_count": 0}))
        records.append(record)
    return records


def _extract_year_value(year: Any, title: Any) -> int | str:
    try:
        if pd.notna(year) and str(year).strip():
            return int(float(str(year).strip()))
    except (ValueError, TypeError):
        pass
    match = re.search(r"\((\d{4})", str(title))
    return int(match.group(1)) if match else ""


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_scalar(value) for key, value in record.items()}


def _json_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
