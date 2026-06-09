"""FastAPI backend for hybrid movie recommendations."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recommender.config import get_settings
from recommender.inference.artifacts import artifact_status
from recommender.inference.artifacts import load_artifact_bundle
from recommender.inference.ratings_store import SidecarRatingStore
from recommender.inference.recommender import HybridArtifactRecommender
from recommender.rag.chatbot import MovieRAGChatbot


app = FastAPI(title="Hybrid Movie Recommendation API", version="0.2.0")


class RecommendationRequest(BaseModel):
    user_id: int | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    session_context: list[str | int] = Field(default_factory=list)
    session_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    exclude_seen: bool = True
    model_name: str = "hybrid"


class MovieRecommendation(BaseModel):
    movie_id: int
    movieId: int | None = None
    tmdb_id: int | None = None
    title: str
    genres: str | None = None
    tmdb_genres: str | None = None
    score: float | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    poster_url: str | None = None
    release_date: str | None = None
    release_year: int | None = None
    year: int | None = None
    overview: str | None = None
    director: str | None = None
    cast: str | None = None
    runtime_minutes: int | None = None
    match_score: float | None = None
    user_rating: float | None = None
    history_source: str | None = None
    explanation_tags: list[str] = Field(default_factory=list)


class RecommendationResponse(BaseModel):
    recommendations: list[MovieRecommendation]


class MoviesResponse(BaseModel):
    movies: list[MovieRecommendation]


class ModelInfoResponse(BaseModel):
    model_info: dict[str, Any]


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(ge=0.5, le=5.0)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_mode: str | None = None


@lru_cache(maxsize=1)
def get_recommender() -> HybridArtifactRecommender:
    settings = get_settings()
    return HybridArtifactRecommender.from_dir(settings.artifacts_dir)


@lru_cache(maxsize=1)
def get_rating_store() -> SidecarRatingStore:
    settings = get_settings()
    return SidecarRatingStore(settings.ratings_store_path)


@lru_cache(maxsize=1)
def get_chatbot() -> MovieRAGChatbot:
    settings = get_settings()
    bundle = load_artifact_bundle(settings.artifacts_dir)
    return MovieRAGChatbot(bundle, api_key=settings.openai_api_key, model=settings.chat_model)


def _service_unavailable(exc: FileNotFoundError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    status = artifact_status(settings.artifacts_dir)
    return {
        "status": "ok" if status["ready"] else "missing_artifacts",
        "artifacts": status,
        "ratings_store_path": str(settings.ratings_store_path),
        "tmdb_api_key_present": bool(settings.tmdb_api_key),
        "openai_api_key_present": bool(settings.openai_api_key),
        "chat_model": settings.chat_model,
    }


@app.get("/users")
def users() -> dict[str, list[int]]:
    try:
        return {"users": get_recommender().users()}
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/users/{user_id}/history", response_model=MoviesResponse)
def user_history(user_id: int, top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        movies = get_recommender().user_history(user_id, rating_store=get_rating_store(), top_k=top_k)
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    return MoviesResponse(movies=movies)


@app.get("/movies", response_model=list[MovieRecommendation])
def movies(
    query: str = Query(default=""),
    search: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        return get_recommender().search_movies(query=search if search is not None else query, limit=limit)
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/movies/trending", response_model=MoviesResponse)
def trending_movies(top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        return MoviesResponse(movies=get_recommender().trending_movies(top_k=top_k))
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/movies/top-rated", response_model=MoviesResponse)
def top_rated_movies(top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        return MoviesResponse(movies=get_recommender().top_rated_movies(top_k=top_k))
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/movies/latest", response_model=MoviesResponse)
def latest_movies(top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        return MoviesResponse(movies=get_recommender().latest_movies(top_k=top_k))
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/movies/genre/{genre}", response_model=MoviesResponse)
def genre_movies(genre: str, top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        return MoviesResponse(movies=get_recommender().genre_movies(genre=genre, top_k=top_k))
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.get("/movies/{movie_id}", response_model=MovieRecommendation)
def movie_details(movie_id: int) -> dict:
    try:
        movie = get_recommender().movie_detail(movie_id)
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    if movie is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy phim")
    return movie


@app.get("/movies/{movie_id}/similar", response_model=MoviesResponse)
def similar_movies(movie_id: int, top_k: int = Query(default=15, ge=1, le=100)) -> MoviesResponse:
    try:
        movies = get_recommender().similar_movies(movie_id, top_k=top_k)
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    return MoviesResponse(movies=movies)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(model_info=get_recommender().model_info())
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc


@app.post("/rate")
def submit_rating(request: RatingRequest) -> dict[str, Any]:
    try:
        if get_recommender().movie_detail(request.movie_id) is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy phim")
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    row = get_rating_store().append(request.user_id, request.movie_id, request.rating)
    return {"status": "success", "message": "Đã lưu đánh giá thành công.", "rating": row}


@app.get("/rate/{user_id}/{movie_id}")
def check_user_rating(user_id: int, movie_id: int) -> dict[str, Any]:
    return {"rating": get_rating_store().latest_rating(user_id, movie_id)}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    try:
        results = get_recommender().recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            session_context=request.session_context,
            session_weight=request.session_weight,
            exclude_seen=request.exclude_seen,
            model_name=request.model_name,
        )
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    return RecommendationResponse(recommendations=results)


@app.post("/recommend", response_model=RecommendationResponse)
def recommend_alias(request: RecommendationRequest) -> RecommendationResponse:
    return recommendations(request)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = get_chatbot().answer(message=request.message, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise _service_unavailable(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Không tạo được câu trả lời chatbot: {exc}") from exc
    return ChatResponse(**result)
