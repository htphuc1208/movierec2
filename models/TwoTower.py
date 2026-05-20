from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EncodedItems:
    movie_ids: list[int]
    vectors: np.ndarray


class MetadataEncoder:
    """Item tower for metadata features with SBERT when available and TF-IDF fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._sbert = None
        self._tfidf: TfidfVectorizer | None = None

    def fit_transform(self, movies: pd.DataFrame) -> EncodedItems:
        texts = self._movie_texts(movies)
        movie_ids = movies["movieId"].astype(int).tolist()
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
        try:
            from sentence_transformers import SentenceTransformer

            self._sbert = SentenceTransformer(self.model_name)
            return self._sbert.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        except ImportError:
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = self._tfidf.fit_transform(texts).astype(np.float32)
            return matrix.toarray()

    @staticmethod
    def _movie_texts(movies: pd.DataFrame) -> list[str]:
        texts: list[str] = []
        for row in movies.itertuples():
            parts = [
                str(getattr(row, "title", "")),
                str(getattr(row, "genres", "")).replace("|", " "),
                str(getattr(row, "overview", "")),
                str(getattr(row, "tagline", "")),
                str(getattr(row, "director", "")),
                str(getattr(row, "cast", "")).replace("|", " "),
            ]
            texts.append(" ".join(parts))
        return texts
