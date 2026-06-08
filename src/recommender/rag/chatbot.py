"""Vietnamese movie RAG chatbot backed by artifact metadata."""

from __future__ import annotations

import os
from typing import Any

from recommender.inference.artifacts import ArtifactBundle
from recommender.rag.retriever import MovieRAGRetriever, RetrievedMovie


class MovieRAGChatbot:
    """Answer movie questions with retrieved catalog context.

    If OPENAI_API_KEY is absent or the OpenAI package is unavailable, the chatbot
    returns a deterministic local answer from retrieved movies. This keeps the
    demo usable without external LLM credentials.
    """

    def __init__(self, bundle: ArtifactBundle, api_key: str | None = None, model: str | None = None) -> None:
        self.retriever = MovieRAGRetriever(bundle)
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("CHAT_MODEL", "gpt-4.1-mini")

    def answer(self, message: str, top_k: int = 6) -> dict[str, Any]:
        message = message.strip()
        movies = self.retriever.retrieve(message, top_k=top_k)
        if not movies:
            return {
                "answer": "Mình chưa tìm thấy phim phù hợp trong catalog hiện tại.",
                "sources": [],
                "retrieval_mode": self.retriever.mode,
            }
        answer = self._llm_answer(message, movies) if self.api_key else self._local_answer(message, movies)
        return {
            "answer": answer,
            "sources": [self._source_payload(movie) for movie in movies],
            "retrieval_mode": self.retriever.mode,
        }

    def _llm_answer(self, message: str, movies: list[RetrievedMovie]) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return self._local_answer(message, movies, note="Chưa cài package openai nên dùng câu trả lời local.")

        context = "\n\n".join(self._movie_context(movie.metadata, idx + 1) for idx, movie in enumerate(movies))
        prompt = f"""Bạn là chatbot tư vấn phim tiếng Việt cho hệ thống gợi ý phim.

Chỉ dùng thông tin trong CONTEXT để trả lời.
Không bịa thông tin ngoài dữ liệu.
Nếu dữ liệu thiếu, hãy nói ngắn gọn là hệ thống chưa có đủ thông tin.
Hãy gợi ý phim phù hợp, giải thích vì sao, giọng tự nhiên.

CONTEXT:
{context}

CÂU HỎI NGƯỜI DÙNG:
{message}
"""
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý gợi ý phim, trả lời bằng tiếng Việt."},
                {"role": "user", "content": prompt},
            ],
        )
        return str(response.choices[0].message.content or "").strip()

    def _local_answer(self, message: str, movies: list[RetrievedMovie], note: str | None = None) -> str:
        lines = []
        if note:
            lines.append(note)
        lines.append("Dựa trên catalog hiện tại, các phim phù hợp nhất là:")
        for idx, movie in enumerate(movies[:5], start=1):
            metadata = movie.metadata
            reason_parts = []
            genres = _clean_text(metadata.get("tmdb_genres") or metadata.get("genres"))
            director = _clean_text(metadata.get("director"))
            overview = _clean_text(metadata.get("overview"))
            if genres:
                reason_parts.append(f"thể loại {genres.replace('|', ', ')}")
            if director:
                reason_parts.append(f"đạo diễn {director.split('|')[0]}")
            if overview:
                reason_parts.append(overview[:140] + ("..." if len(overview) > 140 else ""))
            reason = "; ".join(reason_parts) if reason_parts else "gần nhất với truy vấn theo metadata"
            lines.append(f"{idx}. {movie.title}: {reason}.")
        return "\n".join(lines)

    def _source_payload(self, movie: RetrievedMovie) -> dict[str, Any]:
        metadata = movie.metadata
        return {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "score": movie.score,
            "poster_url": metadata.get("poster_url"),
            "tmdb_id": metadata.get("tmdb_id"),
            "genres": metadata.get("tmdb_genres") or metadata.get("genres"),
            "overview": metadata.get("overview"),
        }

    def _movie_context(self, movie: dict[str, Any], index: int) -> str:
        fields = [
            f"{index}. Title: {movie.get('title', '')}",
            f"Genres: {movie.get('genres', '')}",
            f"TMDb genres: {movie.get('tmdb_genres', '')}",
            f"Overview: {movie.get('overview', '')}",
            f"Director: {movie.get('director', '')}",
            f"Cast: {movie.get('cast', '')}",
            f"Keywords: {movie.get('keywords', '')}",
            f"Release year: {movie.get('release_year', '')}",
            f"Runtime: {movie.get('runtime_minutes', '')}",
        ]
        return "\n".join(field for field in fields if field and not field.endswith(": None"))


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"none", "nan"} else text
