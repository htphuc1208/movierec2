from __future__ import annotations

import os

from openai import OpenAI

from recommender.inference.artifacts import ArtifactBundle
from recommender.rag.retriever import MovieRAGRetriever


class MovieRAGChatbot:
    def __init__(self, bundle: ArtifactBundle) -> None:
        self.retriever = MovieRAGRetriever(bundle)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

    def answer(self, message: str, top_k: int = 6) -> dict:
        movies = self.retriever.retrieve(message, top_k=top_k)

        context = "\n\n".join(
            self._movie_context(movie.metadata, idx + 1)
            for idx, movie in enumerate(movies)
        )

        prompt = f"""
Bạn là chatbot tư vấn phim tiếng Việt cho một hệ thống gợi ý phim.

Chỉ dùng thông tin trong CONTEXT để trả lời.
Không bịa thông tin ngoài dữ liệu.
Nếu dữ liệu thiếu, hãy nói ngắn gọn là hệ thống chưa có đủ thông tin.
Hãy gợi ý phim phù hợp, giải thích vì sao, giọng tự nhiên.

CONTEXT:
{context}

CÂU HỎI NGƯỜI DÙNG:
{message}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là trợ lý gợi ý phim, trả lời bằng tiếng Việt.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [
                {
                    "movie_id": movie.movie_id,
                    "title": movie.title,
                    "score": movie.score,
                    "poster_url": movie.metadata.get("poster_url"),
                    "tmdb_id": movie.metadata.get("tmdb_id"),
                }
                for movie in movies
            ],
        }

    def _movie_context(self, movie: dict, index: int) -> str:
        fields = [
            f"{index}. Title: {movie.get('title', '')}",
            f"Genres: {movie.get('genres', '')}",
            f"TMDB genres: {movie.get('tmdb_genres', '')}",
            f"Overview: {movie.get('overview', '')}",
            f"Director: {movie.get('director', '')}",
            f"Cast: {movie.get('cast', '')}",
            f"Keywords: {movie.get('keywords', '')}",
        ]
        return "\n".join(str(field) for field in fields if field)
