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
def movies() -> dict[str, list[dict[str, Any]]]:
    return {"movies": get_recommender().movies_for_picker()}


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
