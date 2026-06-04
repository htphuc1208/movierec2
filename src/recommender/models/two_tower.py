"""Content-driven two-tower utilities backed by SBERT embeddings."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


EmbeddingBackend = Literal["sbert", "tfidf", "auto"]


def build_item_text(catalog: pd.DataFrame) -> list[str]:
    """Build a text document per movie from enriched metadata."""
    documents: list[str] = []
    for row in catalog.fillna("").itertuples(index=False):
        values = row._asdict()
        parts = [
            str(values.get("title", "")),
            str(values.get("genres", "")),
            str(values.get("tmdb_genres", "")),
            str(values.get("overview", "")),
            str(values.get("tagline", "")),
            str(values.get("director", "")),
            str(values.get("cast", "")),
        ]
        documents.append(" . ".join(part for part in parts if part))
    return documents


def encode_item_texts(
    catalog: pd.DataFrame,
    backend: EmbeddingBackend = "sbert",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    fallback_dim: int = 256,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode item metadata into L2-normalized dense vectors."""
    documents = build_item_text(catalog)
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401

            backend = "sbert"
        except ImportError:
            backend = "tfidf"

    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("SBERT backend requires sentence-transformers. Install requirements.txt first.") from exc
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents)
    max_components = max(1, min(fallback_dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
    if max_components < 2:
        dense = matrix.toarray().astype(np.float32)
    else:
        dense = TruncatedSVD(n_components=max_components, random_state=42).fit_transform(matrix).astype(np.float32)
    return normalize(dense).astype(np.float32)


def build_user_profiles(train_interactions: pd.DataFrame, item_embeddings: np.ndarray, num_users: int) -> np.ndarray:
    """Average positive item embeddings into user preference vectors."""
    dim = item_embeddings.shape[1]
    profiles = np.zeros((num_users, dim), dtype=np.float32)
    counts = np.zeros(num_users, dtype=np.float32)

    for row in train_interactions[["user_idx", "item_idx"]].itertuples(index=False):
        profiles[int(row.user_idx)] += item_embeddings[int(row.item_idx)]
        counts[int(row.user_idx)] += 1.0

    nonzero = counts > 0
    profiles[nonzero] /= counts[nonzero, None]
    return normalize(profiles).astype(np.float32)


def cosine_score_matrix(user_profiles: np.ndarray, item_embeddings: np.ndarray) -> np.ndarray:
    return (user_profiles @ item_embeddings.T).astype(np.float32)
