from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from recommender.inference.artifacts import ArtifactBundle


@dataclass
class RetrievedMovie:
    movie_id: int
    title: str
    score: float
    metadata: dict


class MovieRAGRetriever:
    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.catalog = bundle.catalog.reset_index(drop=True)
        model_name = bundle.hybrid_config.get(
            "sbert_model",
            "sentence-transformers/all-mpnet-base-v2",
        )
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedMovie]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        item_embeddings = normalize(self.bundle.content_embeddings)
        scores = query_embedding @ item_embeddings.T
        scores = scores[0]

        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedMovie] = []
        for idx in top_indices:
            row = self.catalog.iloc[int(idx)]
            metadata = row.to_dict()
            results.append(
                RetrievedMovie(
                    movie_id=int(row.get("movieId", idx)),
                    title=str(row.get("title", "")),
                    score=float(scores[idx]),
                    metadata=metadata,
                )
            )
        return results