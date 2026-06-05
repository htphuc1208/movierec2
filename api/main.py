"""FastAPI backend for hybrid movie recommendations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from openai import AuthenticationError, OpenAIError
from pydantic import BaseModel, Field

from recommender.config import get_settings
from recommender.inference.artifacts import artifact_status
from recommender.inference.recommender import HybridArtifactRecommender
from recommender.inference.artifacts import load_artifact_bundle
from recommender.rag.chatbot import MovieRAGChatbot

app = FastAPI(title="Hybrid Movie Recommendation API", version="0.1.0")


class RecommendationRequest(BaseModel):
    user_id: int | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    session_context: list[str | int] = Field(default_factory=list)


class MovieRecommendation(BaseModel):
    movie_id: int
    tmdb_id: int | None = None
    title: str
    score: float | None = None
    poster_url: str | None = None
    explanation_tags: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    recommendations: list[MovieRecommendation]
    
class ChatRequest(BaseModel):
    message: str
    top_k: int = Field(default=6, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


@lru_cache(maxsize=1)
def get_recommender() -> HybridArtifactRecommender:
    settings = get_settings()
    return HybridArtifactRecommender.from_dir(settings.artifacts_dir)

@lru_cache(maxsize=1)
def get_chatbot() -> MovieRAGChatbot:
    settings = get_settings()
    bundle = load_artifact_bundle(settings.artifacts_dir)
    return MovieRAGChatbot(bundle)


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    status = artifact_status(settings.artifacts_dir)
    return {
        "status": "ok" if status["ready"] else "missing_artifacts",
        "artifacts": status,
        "tmdb_api_key_present": bool(settings.tmdb_api_key),
    }


@app.get("/movies", response_model=list[MovieRecommendation])
def movies(query: str = Query(default=""), limit: int = Query(default=20, ge=1, le=100)):
    try:
        return get_recommender().search_movies(query=query, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    try:
        results = get_recommender().recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            session_context=request.session_context,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return RecommendationResponse(recommendations=results)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = get_chatbot().answer(
            message=request.message,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="OPENAI_API_KEY không hợp lệ. Hãy thay key thật trong .env rồi restart API.",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    return ChatResponse(**result)
