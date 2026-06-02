from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    F = None


@dataclass
class EncodedItems:
    movie_ids: list[int]
    vectors: np.ndarray


@dataclass
class ContentRecommendation:
    movie_id: int
    title: str
    genres: str
    score: float


class MetadataEncoder:
    """Item tower for metadata features with SBERT when available and TF-IDF fallback."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: Literal["auto", "tfidf", "sbert"] = "auto",
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self._sbert = None
        self._tfidf: TfidfVectorizer | None = None

    def fit_transform(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> EncodedItems:
        normalised = self._normalise_movies(movies, tags)
        texts = self._movie_texts(normalised)
        movie_ids = normalised["movieId"].astype(int).tolist()
        vectors = self._encode_texts(texts)
        return EncodedItems(movie_ids=movie_ids, vectors=vectors)

    def user_profile(self, item_vectors: np.ndarray, item_indices: list[int]) -> np.ndarray:
        if not item_indices:
            return np.zeros(item_vectors.shape[1], dtype=np.float32)
        return item_vectors[item_indices].mean(axis=0)

    def score_items(self, profile: np.ndarray, item_vectors: np.ndarray) -> np.ndarray:
        if profile.ndim == 1:
            profile = profile.reshape(1, -1)
        return cosine_similarity(profile, item_vectors).ravel()

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        if self.backend in {"auto", "sbert"}:
            try:
                from sentence_transformers import SentenceTransformer

                self._sbert = SentenceTransformer(self.model_name)
                return self._sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            except Exception:
                if self.backend == "sbert":
                    raise

        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = self._tfidf.fit_transform(texts).astype(np.float32)
        return matrix.toarray()

    @staticmethod
    def _normalise_movies(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.DataFrame:
        normalised = movies.copy().reset_index(drop=True)
        if "movieId" not in normalised.columns and "movie_id" in normalised.columns:
            normalised = normalised.rename(columns={"movie_id": "movieId"})
        if "movieId" not in normalised.columns:
            raise ValueError("movies must contain movieId or movie_id")

        for column in [
            "title",
            "genres",
            "overview",
            "tagline",
            "director",
            "cast",
            "tags",
            "keywords",
            "tag_genome_tags",
            "original_language",
            "production_countries",
            "collection_name",
            "certification",
        ]:
            if column not in normalised.columns:
                normalised[column] = ""
            normalised[column] = normalised[column].fillna("").astype(str)

        if tags is not None and not tags.empty:
            tag_frame = tags.copy()
            if "movieId" not in tag_frame.columns and "movie_id" in tag_frame.columns:
                tag_frame = tag_frame.rename(columns={"movie_id": "movieId"})
            if "tag" in tag_frame.columns and "movieId" in tag_frame.columns:
                joined_tags = tag_frame.groupby("movieId")["tag"].apply(lambda values: " ".join(str(value) for value in values))
                normalised["tags"] = normalised["movieId"].map(joined_tags).fillna(normalised["tags"])

        return normalised

    @classmethod
    def _movie_texts(cls, movies: pd.DataFrame) -> list[str]:
        movies = cls._normalise_movies(movies)
        texts: list[str] = []
        for row in movies.itertuples():
            parts = [
                str(getattr(row, "title", "")),
                str(getattr(row, "genres", "")).replace("|", " "),
                str(getattr(row, "tags", "")),
                str(getattr(row, "overview", "")),
                str(getattr(row, "tagline", "")),
                str(getattr(row, "director", "")),
                str(getattr(row, "cast", "")).replace("|", " "),
                str(getattr(row, "keywords", "")).replace("|", " "),
                str(getattr(row, "tag_genome_tags", "")).replace("|", " "),
                str(getattr(row, "original_language", "")),
                str(getattr(row, "production_countries", "")).replace("|", " "),
                str(getattr(row, "collection_name", "")),
                str(getattr(row, "certification", "")),
            ]
            texts.append(" ".join(parts))
        return texts


class ContentRecommender:
    """Content-only recommender for item similarity and cold-start user profiles."""

    def __init__(
        self,
        backend: Literal["auto", "tfidf", "sbert"] = "tfidf",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifact_dir: str | Path = "artifacts",
    ) -> None:
        self.encoder = MetadataEncoder(model_name=model_name, backend=backend)
        self.artifact_dir = Path(artifact_dir)
        self.movies: pd.DataFrame | None = None
        self.movie_ids: list[int] = []
        self.movie_index: dict[int, int] = {}
        self.item_vectors: np.ndarray | None = None
        self.similarity_matrix: np.ndarray | None = None

    def fit(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None, save_artifacts: bool = False) -> "ContentRecommender":
        self.movies = MetadataEncoder._normalise_movies(movies, tags)
        encoded = self.encoder.fit_transform(self.movies)
        self.movie_ids = encoded.movie_ids
        self.movie_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}
        self.item_vectors = encoded.vectors
        self.similarity_matrix = cosine_similarity(self.item_vectors, self.item_vectors)

        if save_artifacts:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.artifact_dir / "content_item_vectors.npy", self.item_vectors)
            np.save(self.artifact_dir / "content_similarity.npy", self.similarity_matrix)

        return self

    def recommend_similar_movies(self, movie_id: int, top_k: int = 10) -> list[dict[str, object]]:
        self._ensure_fit()
        idx = self.movie_index.get(int(movie_id))
        if idx is None:
            return []
        scores = self.similarity_matrix[idx].copy()
        scores[idx] = -np.inf
        ranked = np.argsort(scores)[::-1]
        return self._records_from_indices(ranked[:top_k], scores)

    def recommend_for_user(
        self,
        user_id: int,
        user_history: pd.DataFrame,
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        self._ensure_fit()
        history = user_history.copy()
        if "userId" not in history.columns and "user_id" in history.columns:
            history = history.rename(columns={"user_id": "userId"})
        if "movieId" not in history.columns and "movie_id" in history.columns:
            history = history.rename(columns={"movie_id": "movieId"})
        if "userId" not in history.columns or "movieId" not in history.columns:
            raise ValueError("user_history must contain userId/movieId or user_id/movie_id")

        watched_movie_ids = history.loc[history["userId"].astype(int) == int(user_id), "movieId"].astype(int).tolist()
        watched_indices = [self.movie_index[movie_id] for movie_id in watched_movie_ids if movie_id in self.movie_index]
        if not watched_indices:
            return []

        profile = self.encoder.user_profile(self.item_vectors, watched_indices)
        scores = self.encoder.score_items(profile, self.item_vectors)
        scores[watched_indices] = -np.inf
        ranked = np.argsort(scores)[::-1]
        return self._records_from_indices(ranked[:top_k], scores)

    def _records_from_indices(self, indices: np.ndarray, scores: np.ndarray) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for idx in indices:
            if not np.isfinite(scores[int(idx)]):
                continue
            movie = self.movies.iloc[int(idx)]
            records.append(
                ContentRecommendation(
                    movie_id=int(movie["movieId"]),
                    title=str(movie.get("title", "")),
                    genres=str(movie.get("genres", "")),
                    score=float(scores[int(idx)]),
                ).__dict__
            )
        return records

    def _ensure_fit(self) -> None:
        if self.movies is None or self.item_vectors is None or self.similarity_matrix is None:
            raise RuntimeError("ContentRecommender.fit must be called before inference.")


class TFIDFRecommender(ContentRecommender):
    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        super().__init__(backend="tfidf", artifact_dir=artifact_dir)


class SBERTRecommender(ContentRecommender):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifact_dir: str | Path = "artifacts",
    ) -> None:
        super().__init__(backend="sbert", model_name=model_name, artifact_dir=artifact_dir)


if torch is not None:

    class TwoTowerModel(nn.Module):
        """Small neural two-tower projection for user/item metadata vectors."""

        def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.2) -> None:
            super().__init__()
            self.user_tower = self._tower(input_dim, hidden_dim, output_dim, dropout)
            self.item_tower = self._tower(input_dim, hidden_dim, output_dim, dropout)

        @staticmethod
        def _tower(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
            user_embeddings = F.normalize(self.user_tower(user_features), p=2, dim=1)
            item_embeddings = F.normalize(self.item_tower(item_features), p=2, dim=1)
            return torch.sum(user_embeddings * item_embeddings, dim=1)

else:

    class TwoTowerModel:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("TwoTowerModel requires torch. Install requirements-ml.txt.")
