from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    collaborative_score: float
    content_score: float
    popularity_score: float
    genres: str
    year: str | int
    tmdb_id: str
    poster_url: str
    overview: str
    reason: list[str]


class HybridMovieRecommender:
    """A lightweight SVD + TF-IDF hybrid recommender for API/UI inference."""

    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.35,
        popularity_weight: float = 0.10,
        min_rating: float = 4.0,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.popularity_weight = popularity_weight
        self.min_rating = min_rating
        self.movies: pd.DataFrame | None = None
        self.ratings: pd.DataFrame | None = None
        self.tags: pd.DataFrame | None = None
        self.movie_ids: list[int] = []
        self.user_ids: list[int] = []
        self.movie_index: dict[int, int] = {}
        self.user_index: dict[int, int] = {}
        self._rating_matrix: np.ndarray | None = None
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._content_matrix: Any = None
        self._vectorizer: TfidfVectorizer | None = None
        self._popularity: np.ndarray | None = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame | None = None) -> "HybridMovieRecommender":
        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.tags = tags.copy() if tags is not None else pd.DataFrame(columns=["movieId", "tag"])

        self.movie_ids = self.movies["movieId"].astype(int).tolist()
        self.user_ids = sorted(self.ratings["userId"].astype(int).unique().tolist())
        self.movie_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}

        self._build_collaborative_space()
        self._build_content_space()
        self._build_popularity()
        return self

    def recommend(
        self,
        user_id: int | None = None,
        top_k: int = 10,
        session_context: list[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_fit()
        session_movie_ids = self.resolve_movie_tokens(session_context or [])
        cf_scores = self._collaborative_scores(user_id)
        content_scores = self._content_scores(user_id, session_movie_ids)
        popularity_scores = self._popularity_scores()

        scores = (
            self.alpha * self._scale_scores(cf_scores)
            + self.beta * self._scale_scores(content_scores)
            + self.popularity_weight * self._scale_scores(popularity_scores)
        )

        excluded = set(session_movie_ids)
        if exclude_seen and user_id is not None:
            excluded.update(self.seen_movies(user_id))
        for movie_id in excluded:
            idx = self.movie_index.get(movie_id)
            if idx is not None:
                scores[idx] = -np.inf

        candidate_indices = np.argsort(scores)[::-1]
        recommendations: list[dict[str, Any]] = []
        for idx in candidate_indices:
            if len(recommendations) >= top_k:
                break
            if not np.isfinite(scores[idx]):
                continue
            movie = self.movies.iloc[idx]
            rec = Recommendation(
                movie_id=int(movie["movieId"]),
                title=str(movie["title"]),
                score=float(scores[idx]),
                collaborative_score=float(cf_scores[idx]),
                content_score=float(content_scores[idx]),
                popularity_score=float(popularity_scores[idx]),
                genres=str(movie.get("genres", "")),
                year=self._json_scalar(movie.get("year", "")),
                tmdb_id=str(movie.get("tmdbId", "")),
                poster_url=str(movie.get("poster_url", "")),
                overview=str(movie.get("overview", "")),
                reason=self._reason_for(movie, user_id, session_movie_ids, cf_scores[idx], content_scores[idx]),
            )
            recommendations.append(rec.__dict__)
        return recommendations

    def seen_movies(self, user_id: int) -> set[int]:
        if self.ratings is None:
            return set()
        return set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"].astype(int).tolist())

    def resolve_movie_tokens(self, tokens: list[str]) -> list[int]:
        self._ensure_fit()
        resolved: list[int] = []
        by_title = {str(row.title).lower(): int(row.movieId) for row in self.movies.itertuples()}
        by_tmdb = {}
        if "tmdbId" in self.movies.columns:
            for row in self.movies.itertuples():
                tmdb_id = getattr(row, "tmdbId", "")
                if pd.notna(tmdb_id) and str(tmdb_id):
                    by_tmdb[f"tmdb_{str(tmdb_id).split('.')[0]}"] = int(row.movieId)

        for token in tokens:
            value = str(token).strip()
            if not value:
                continue
            movie_id: int | None = None
            if value.isdigit():
                movie_id = int(value)
            elif value.lower() in by_tmdb:
                movie_id = by_tmdb[value.lower()]
            elif value.lower() in by_title:
                movie_id = by_title[value.lower()]
            if movie_id in self.movie_index and movie_id not in resolved:
                resolved.append(movie_id)
        return resolved

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        self._ensure_fit()
        if movie_id not in self.movie_index:
            return 3.0
        idx = self.movie_index[movie_id]
        cf = self._collaborative_scores(user_id)[idx]
        content = self._content_scores(user_id, [])[idx]
        popularity = self._popularity_scores()[idx]
        raw = 0.65 * self._rating_like(cf) + 0.25 * self._rating_like(content) + 0.10 * self._rating_like(popularity)
        return float(np.clip(raw, 0.5, 5.0))

    def users(self) -> list[int]:
        return list(self.user_ids)

    def movies_for_picker(self) -> list[dict[str, Any]]:
        self._ensure_fit()
        fields = ["movieId", "title", "genres", "year", "tmdbId", "poster_url"]
        existing = [field for field in fields if field in self.movies.columns]
        records = self.movies[existing].to_dict(orient="records")
        return [{key: self._json_scalar(value) for key, value in record.items()} for record in records]

    def _build_collaborative_space(self) -> None:
        user_count = len(self.user_ids)
        item_count = len(self.movie_ids)
        matrix = np.zeros((user_count, item_count), dtype=np.float32)
        for row in self.ratings.itertuples():
            user_idx = self.user_index.get(int(row.userId))
            item_idx = self.movie_index.get(int(row.movieId))
            if user_idx is not None and item_idx is not None:
                matrix[user_idx, item_idx] = float(row.rating)
        self._rating_matrix = matrix

        if user_count >= 2 and item_count >= 2 and np.count_nonzero(matrix) > 0:
            n_components = max(1, min(16, user_count - 1, item_count - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            self._user_factors = svd.fit_transform(matrix)
            self._item_factors = svd.components_.T
        else:
            self._user_factors = np.zeros((user_count, 1), dtype=np.float32)
            self._item_factors = np.zeros((item_count, 1), dtype=np.float32)

    def _build_content_space(self) -> None:
        tag_map = self._tags_by_movie()
        text = []
        for row in self.movies.itertuples():
            movie_id = int(row.movieId)
            parts = [
                str(getattr(row, "title", "")),
                str(getattr(row, "genres", "")).replace("|", " "),
                str(getattr(row, "overview", "")),
                str(getattr(row, "tagline", "")),
                str(getattr(row, "director", "")),
                str(getattr(row, "cast", "")).replace("|", " "),
                " ".join(tag_map.get(movie_id, [])),
            ]
            text.append(" ".join(parts))
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._content_matrix = self._vectorizer.fit_transform(text)

    def _build_popularity(self) -> None:
        grouped = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        popularity = np.zeros(len(self.movie_ids), dtype=np.float32)
        max_count = max(float(grouped["count"].max()), 1.0) if not grouped.empty else 1.0
        for movie_id, row in grouped.iterrows():
            idx = self.movie_index.get(int(movie_id))
            if idx is not None:
                popularity[idx] = (float(row["mean"]) / 5.0) * np.log1p(float(row["count"])) / np.log1p(max_count)
        self._popularity = popularity

    def _collaborative_scores(self, user_id: int | None) -> np.ndarray:
        self._ensure_fit()
        if user_id is not None and user_id in self.user_index:
            user_idx = self.user_index[user_id]
            scores = self._user_factors[user_idx] @ self._item_factors.T
            return scores.astype(np.float32)
        return self._popularity_scores()

    def _content_scores(self, user_id: int | None, session_movie_ids: list[int]) -> np.ndarray:
        self._ensure_fit()
        profile_ids = list(session_movie_ids)
        if user_id is not None:
            liked = self.ratings.loc[
                (self.ratings["userId"] == user_id) & (self.ratings["rating"] >= self.min_rating),
                "movieId",
            ].astype(int)
            profile_ids.extend([movie_id for movie_id in liked.tolist() if movie_id not in profile_ids])

        profile_indices = [self.movie_index[movie_id] for movie_id in profile_ids if movie_id in self.movie_index]
        if not profile_indices:
            return self._popularity_scores()

        profile_vector = np.asarray(self._content_matrix[profile_indices].mean(axis=0)).reshape(1, -1)
        scores = cosine_similarity(profile_vector, self._content_matrix).ravel()
        return scores.astype(np.float32)

    def _popularity_scores(self) -> np.ndarray:
        self._ensure_fit()
        return self._popularity.astype(np.float32)

    def _reason_for(
        self,
        movie: pd.Series,
        user_id: int | None,
        session_movie_ids: list[int],
        cf_score: float,
        content_score: float,
    ) -> list[str]:
        reasons: list[str] = []
        if content_score > 0.15:
            shared = self._shared_genres(movie, user_id, session_movie_ids)
            if shared:
                reasons.append(f"Similar genres: {', '.join(shared[:3])}")
            else:
                reasons.append("Similar metadata profile")
        if cf_score > 0:
            reasons.append("Liked by users with related taste")
        director = str(movie.get("director", "")).split("|")[0].strip()
        if director:
            reasons.append(f"Director: {director}")
        if not reasons:
            reasons.append("High rating popularity")
        return reasons[:3]

    def _shared_genres(self, movie: pd.Series, user_id: int | None, session_movie_ids: list[int]) -> list[str]:
        target = set(str(movie.get("genres", "")).split("|"))
        profile_ids = list(session_movie_ids)
        if user_id is not None and self.ratings is not None:
            profile_ids.extend(
                self.ratings.loc[
                    (self.ratings["userId"] == user_id) & (self.ratings["rating"] >= self.min_rating),
                    "movieId",
                ].astype(int).tolist()
            )
        profile_genres: set[str] = set()
        for movie_id in profile_ids:
            idx = self.movie_index.get(movie_id)
            if idx is not None:
                profile_genres.update(str(self.movies.iloc[idx].get("genres", "")).split("|"))
        return sorted(genre for genre in target.intersection(profile_genres) if genre)

    def _tags_by_movie(self) -> dict[int, list[str]]:
        if self.tags is None or self.tags.empty or "tag" not in self.tags.columns:
            return {}
        grouped = self.tags.groupby("movieId")["tag"].apply(lambda values: [str(value) for value in values])
        return {int(movie_id): values for movie_id, values in grouped.items()}

    @staticmethod
    def _scale_scores(scores: np.ndarray) -> np.ndarray:
        finite = np.isfinite(scores)
        if not finite.any():
            return np.zeros_like(scores, dtype=np.float32)
        safe = scores.copy().astype(np.float32)
        safe[~finite] = np.nanmin(safe[finite])
        if np.nanmax(safe) == np.nanmin(safe):
            return np.zeros_like(safe, dtype=np.float32)
        return MinMaxScaler().fit_transform(safe.reshape(-1, 1)).ravel().astype(np.float32)

    @staticmethod
    def _rating_like(score: float) -> float:
        return 0.5 + 4.5 * float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if pd.isna(value):
            return ""
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _ensure_fit(self) -> None:
        if self.movies is None or self.ratings is None:
            raise RuntimeError("HybridMovieRecommender.fit must be called before inference.")
