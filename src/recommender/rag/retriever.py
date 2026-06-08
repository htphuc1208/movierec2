"""Movie retrieval for the RAG chatbot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recommender.eval.metrics import top_k_from_scores
from recommender.inference.artifacts import ArtifactBundle
from recommender.models.two_tower import build_item_text


@dataclass(frozen=True)
class RetrievedMovie:
    movie_id: int
    title: str
    score: float
    metadata: dict[str, Any]


class MovieRAGRetriever:
    """Retrieve movies from exported artifacts using semantic or lexical search."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.catalog = bundle.catalog.reset_index(drop=True).copy()
        self.documents = build_item_text(self.catalog)
        self.content_backend = str(bundle.hybrid_config.get("content_backend", "")).lower()
        self.sbert_model_name = str(bundle.hybrid_config.get("sbert_model", "sentence-transformers/all-mpnet-base-v2"))
        self.item_embeddings = normalize(bundle.content_embeddings.astype(np.float32))
        self._sbert_model = None
        self._sbert_error = ""
        self._lexical_vectorizer: TfidfVectorizer | None = None
        self._lexical_matrix = None

    @property
    def mode(self) -> str:
        if self.content_backend == "sbert" and not self._sbert_error:
            return "semantic"
        return "lexical"

    @property
    def semantic_error(self) -> str:
        return self._sbert_error

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedMovie]:
        query = query.strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), 50))
        scores = self._semantic_scores(query)
        if scores is None:
            scores = self._lexical_scores(query)
        top_indices = top_k_from_scores(scores.astype(np.float32), top_k)
        return [self._result_from_item(idx, float(scores[idx])) for idx in top_indices if np.isfinite(scores[idx])]

    def _semantic_scores(self, query: str) -> np.ndarray | None:
        if self.content_backend != "sbert":
            return None
        try:
            model = self._load_sbert_model()
            query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        except Exception as exc:
            self._sbert_error = str(exc)
            return None
        if query_embedding.shape[1] != self.item_embeddings.shape[1]:
            self._sbert_error = f"query dim {query_embedding.shape[1]} != artifact dim {self.item_embeddings.shape[1]}"
            return None
        return (query_embedding @ self.item_embeddings.T)[0].astype(np.float32)

    def _load_sbert_model(self):
        if self._sbert_model is not None:
            return self._sbert_model
        from sentence_transformers import SentenceTransformer

        self._sbert_model = SentenceTransformer(self.sbert_model_name)
        return self._sbert_model

    def _lexical_scores(self, query: str) -> np.ndarray:
        if self._lexical_vectorizer is None or self._lexical_matrix is None:
            self._lexical_vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=1)
            self._lexical_matrix = self._lexical_vectorizer.fit_transform(self.documents)
        query_vector = self._lexical_vectorizer.transform([query])
        return (query_vector @ self._lexical_matrix.T).toarray()[0].astype(np.float32)

    def _result_from_item(self, item_idx: int, score: float) -> RetrievedMovie:
        row = self.catalog.iloc[int(item_idx)]
        movie_id = _optional_int(row.get("movieId"))
        metadata = {str(key): _jsonable(value) for key, value in row.to_dict().items()}
        return RetrievedMovie(
            movie_id=movie_id if movie_id is not None else int(item_idx),
            title=str(row.get("title", "")),
            score=score,
            metadata=metadata,
        )


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
