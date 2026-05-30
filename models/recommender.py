from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    """A lightweight Funk-SVD + TF-IDF hybrid recommender for API/UI inference."""

    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.35,
        popularity_weight: float = 0.10,
        min_rating: float = 4.0,
        collaborative_factors: int = 32,
        collaborative_epochs: int = 12,
        collaborative_lr: float = 0.01,
        collaborative_reg: float = 0.02,
        collaborative_batch_size: int = 4096,
        random_state: int = 42,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.popularity_weight = popularity_weight
        self.min_rating = min_rating
        self.collaborative_factors = collaborative_factors
        self.collaborative_epochs = collaborative_epochs
        self.collaborative_lr = collaborative_lr
        self.collaborative_reg = collaborative_reg
        self.collaborative_batch_size = collaborative_batch_size
        self.random_state = random_state
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
        self._user_bias: np.ndarray | None = None
        self._item_bias: np.ndarray | None = None
        self._global_mean: float = 0.0
        self._content_matrix: Any = None
        self._vectorizer: TfidfVectorizer | None = None
        self._popularity: np.ndarray | None = None
        self._collaborative_mode = "funk_svd"
        self.model_source = "unfit"
        self.model_name = "hybrid-funk-svd"
        self.dataset_name = ""
        self.artifact_path = ""
        self.metrics: dict[str, Any] = {}
        self.artifact_manifest: dict[str, Any] = {}

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
        self._collaborative_mode = "funk_svd"
        self.model_source = "runtime_fit"
        self.model_name = "hybrid-funk-svd"
        return self

    def load_artifact(
        self,
        artifact_dir: str | Path,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: pd.DataFrame | None = None,
    ) -> "HybridMovieRecommender":
        artifact_path = Path(artifact_dir)
        manifest_path = artifact_path / "manifest.json"
        collaborative_path = artifact_path / "collaborative.npz"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing recommender manifest: {manifest_path}")
        if not collaborative_path.exists():
            raise FileNotFoundError(f"Missing collaborative artifact: {collaborative_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        weights = manifest.get("weights", {})
        self.alpha = float(weights.get("collaborative", self.alpha))
        self.beta = float(weights.get("content", self.beta))
        self.popularity_weight = float(weights.get("popularity", self.popularity_weight))
        self.min_rating = float(manifest.get("positive_threshold", self.min_rating))

        self.movies = movies.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.tags = tags.copy() if tags is not None else pd.DataFrame(columns=["movieId", "tag"])
        self.movie_ids = self.movies["movieId"].astype(int).tolist()
        self.movie_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}

        data = np.load(collaborative_path, allow_pickle=False)
        artifact_user_ids = data["user_ids"].astype(np.int64).tolist()
        artifact_movie_ids = data["movie_ids"].astype(np.int64).tolist()
        user_embeddings = data["user_embeddings"].astype(np.float32)
        item_embeddings = data["item_embeddings"].astype(np.float32)

        self.user_ids = [int(user_id) for user_id in artifact_user_ids]
        self.user_index = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self._user_factors = user_embeddings
        self._item_factors = self._align_item_embeddings(artifact_movie_ids, item_embeddings)
        self._user_bias = self._load_optional_vector(data, "user_bias", len(self.user_ids))
        self._item_bias = self._align_optional_item_vector(data, "item_bias", artifact_movie_ids)
        self._global_mean = float(data["global_mean"][0]) if "global_mean" in data else float(self.ratings["rating"].mean())
        self._rating_matrix = None
        self._collaborative_mode = str(manifest.get("collaborative", {}).get("mode", "embedding"))

        self._build_content_space()
        self._build_popularity()

        self.model_source = "artifact"
        self.model_name = str(manifest.get("model_name", "artifact-hybrid"))
        self.dataset_name = str(manifest.get("dataset", ""))
        self.artifact_path = str(artifact_path)
        self.metrics = dict(manifest.get("metrics", {}))
        self.artifact_manifest = manifest
        return self

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: pd.DataFrame | None = None,
    ) -> "HybridMovieRecommender":
        return cls().load_artifact(artifact_dir, movies, ratings, tags)

    def save_artifact(
        self,
        artifact_dir: str | Path,
        dataset_name: str,
        model_name: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_fit()
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        collaborative_path = path / "collaborative.npz"

        np.savez_compressed(
            collaborative_path,
            user_ids=np.asarray(self.user_ids, dtype=np.int64),
            movie_ids=np.asarray(self.movie_ids, dtype=np.int64),
            user_embeddings=np.asarray(self._user_factors, dtype=np.float32),
            item_embeddings=np.asarray(self._item_factors, dtype=np.float32),
            user_bias=np.asarray(self._user_bias, dtype=np.float32),
            item_bias=np.asarray(self._item_bias, dtype=np.float32),
            global_mean=np.asarray([self._global_mean], dtype=np.float32),
        )

        manifest: dict[str, Any] = {
            "artifact_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "model_name": model_name or self.model_name,
            "model_source": self.model_source,
            "positive_threshold": self.min_rating,
            "weights": {
                "collaborative": self.alpha,
                "content": self.beta,
                "popularity": self.popularity_weight,
            },
            "collaborative": {
                "mode": self._collaborative_mode,
                "factors": int(self._user_factors.shape[1]) if self._user_factors is not None else 0,
            },
            "content": {"backend": "tfidf"},
            "files": {"collaborative": "collaborative.npz"},
            "metrics": metrics or self.metrics,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

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
        rating = self._collaborative_rating(user_id, movie_id)
        return float(np.clip(rating, 0.5, 5.0))

    def users(self) -> list[int]:
        return list(self.user_ids)

    def movies_for_picker(self) -> list[dict[str, Any]]:
        self._ensure_fit()
        fields = ["movieId", "title", "genres", "year", "tmdbId", "poster_url"]
        existing = [field for field in fields if field in self.movies.columns]
        records = self.movies[existing].to_dict(orient="records")
        return [{key: self._json_scalar(value) for key, value in record.items()} for record in records]

    def model_info(self) -> dict[str, Any]:
        self._ensure_fit()
        return {
            "model_source": self.model_source,
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "artifact_path": self.artifact_path,
            "weights": {
                "collaborative": self.alpha,
                "content": self.beta,
                "popularity": self.popularity_weight,
            },
            "positive_threshold": self.min_rating,
            "collaborative_mode": self._collaborative_mode,
            "metrics": self.metrics,
            "user_count": len(self.user_ids),
            "movie_count": len(self.movie_ids),
        }

    def _build_collaborative_space(self) -> None:
        user_count = len(self.user_ids)
        item_count = len(self.movie_ids)
        factor_count = max(1, min(self.collaborative_factors, max(user_count, 1), max(item_count, 1)))
        rng = np.random.default_rng(self.random_state)
        self._global_mean = float(self.ratings["rating"].mean()) if not self.ratings.empty else 3.0
        self._user_factors = rng.normal(0.0, 0.05, size=(user_count, factor_count)).astype(np.float32)
        self._item_factors = rng.normal(0.0, 0.05, size=(item_count, factor_count)).astype(np.float32)
        self._user_bias = np.zeros(user_count, dtype=np.float32)
        self._item_bias = np.zeros(item_count, dtype=np.float32)

        if user_count == 0 or item_count == 0 or self.ratings.empty:
            self._rating_matrix = np.zeros((user_count, item_count), dtype=np.float32)
            return

        interactions = []
        for row in self.ratings.itertuples():
            user_idx = self.user_index.get(int(row.userId))
            item_idx = self.movie_index.get(int(row.movieId))
            if user_idx is not None and item_idx is not None:
                interactions.append((user_idx, item_idx, float(row.rating)))
        if not interactions:
            self._rating_matrix = np.zeros((user_count, item_count), dtype=np.float32)
            return

        data = np.asarray(interactions, dtype=np.float32)
        user_indices = data[:, 0].astype(np.int64)
        item_indices = data[:, 1].astype(np.int64)
        ratings = data[:, 2].astype(np.float32)
        self._rating_matrix = np.zeros((user_count, item_count), dtype=np.float32)
        self._rating_matrix[user_indices, item_indices] = ratings

        for _ in range(max(0, self.collaborative_epochs)):
            order = rng.permutation(len(ratings))
            for start in range(0, len(order), self.collaborative_batch_size):
                batch = order[start : start + self.collaborative_batch_size]
                users = user_indices[batch]
                items = item_indices[batch]
                labels = ratings[batch]

                user_vecs = self._user_factors[users].copy()
                item_vecs = self._item_factors[items].copy()
                user_bias = self._user_bias[users]
                item_bias = self._item_bias[items]
                preds = self._global_mean + user_bias + item_bias + np.sum(user_vecs * item_vecs, axis=1)
                errors = preds - labels

                np.add.at(
                    self._user_factors,
                    users,
                    -self.collaborative_lr * (errors[:, None] * item_vecs + self.collaborative_reg * user_vecs),
                )
                np.add.at(
                    self._item_factors,
                    items,
                    -self.collaborative_lr * (errors[:, None] * user_vecs + self.collaborative_reg * item_vecs),
                )
                np.add.at(
                    self._user_bias,
                    users,
                    -self.collaborative_lr * (errors + self.collaborative_reg * user_bias),
                )
                np.add.at(
                    self._item_bias,
                    items,
                    -self.collaborative_lr * (errors + self.collaborative_reg * item_bias),
                )

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
            if self._collaborative_mode == "embedding":
                scores = self._user_factors[user_idx] @ self._item_factors.T
                return scores.astype(np.float32)
            scores = (
                self._global_mean
                + self._user_bias[user_idx]
                + self._item_bias
                + self._user_factors[user_idx] @ self._item_factors.T
            )
            return scores.astype(np.float32)
        return self._popularity_scores()

    def _collaborative_rating(self, user_id: int, movie_id: int) -> float:
        item_idx = self.movie_index.get(movie_id)
        if item_idx is None:
            return 3.0
        if self._collaborative_mode == "embedding":
            scores = self._collaborative_scores(user_id)
            scaled = self._scale_scores(scores)
            return self._rating_like(float(scaled[item_idx]))
        item_bias = float(self._item_bias[item_idx]) if self._item_bias is not None else 0.0
        if user_id not in self.user_index:
            return self._global_mean + item_bias
        user_idx = self.user_index[user_id]
        return float(
            self._global_mean
            + self._user_bias[user_idx]
            + self._item_bias[item_idx]
            + self._user_factors[user_idx] @ self._item_factors[item_idx]
        )

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
        if cf_score >= max(self._global_mean, 3.5):
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

    def _align_item_embeddings(self, artifact_movie_ids: list[int], item_embeddings: np.ndarray) -> np.ndarray:
        factor_count = int(item_embeddings.shape[1]) if item_embeddings.ndim == 2 else 1
        aligned = np.zeros((len(self.movie_ids), factor_count), dtype=np.float32)
        artifact_index = {int(movie_id): idx for idx, movie_id in enumerate(artifact_movie_ids)}
        for movie_id, target_idx in self.movie_index.items():
            source_idx = artifact_index.get(int(movie_id))
            if source_idx is not None:
                aligned[target_idx] = item_embeddings[source_idx]
        return aligned

    @staticmethod
    def _load_optional_vector(data: Any, name: str, size: int) -> np.ndarray:
        if name in data:
            values = data[name].astype(np.float32)
            if values.shape[0] == size:
                return values
        return np.zeros(size, dtype=np.float32)

    def _align_optional_item_vector(self, data: Any, name: str, artifact_movie_ids: list[int]) -> np.ndarray:
        if name not in data:
            return np.zeros(len(self.movie_ids), dtype=np.float32)
        values = data[name].astype(np.float32)
        aligned = np.zeros(len(self.movie_ids), dtype=np.float32)
        artifact_index = {int(movie_id): idx for idx, movie_id in enumerate(artifact_movie_ids)}
        for movie_id, target_idx in self.movie_index.items():
            source_idx = artifact_index.get(int(movie_id))
            if source_idx is not None and source_idx < values.shape[0]:
                aligned[target_idx] = values[source_idx]
        return aligned

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
        if self.movies is None or self.ratings is None or self._popularity is None:
            raise RuntimeError("HybridMovieRecommender.fit must be called before inference.")
