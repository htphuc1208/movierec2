from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import MovieLensDataLoader
from models import HybridMovieRecommender


class RecommendRequest(BaseModel):
    user_id: int | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    session_context: list[str] = Field(default_factory=list)
    exclude_seen: bool = True


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]


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
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users")
def users() -> dict[str, list[int]]:
    return {"users": get_recommender().users()}


@app.get("/movies")
def movies() -> dict[str, list[dict[str, Any]]]:
    return {"movies": get_recommender().movies_for_picker()}


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
