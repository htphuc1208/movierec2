"""Hybrid artifact-based recommender used by FastAPI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from recommender.eval.metrics import minmax, top_k_from_scores
from recommender.inference.artifacts import ArtifactBundle, load_artifact_bundle


class HybridArtifactRecommender:
    """Serve top-K recommendations from exported model artifacts."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle
        self.catalog = bundle.catalog.reset_index(drop=True).copy()
        self.catalog["item_idx"] = np.arange(len(self.catalog))
        self.item_mapping = bundle.item_mapping
        self.user_mapping = bundle.user_mapping
        self.index_to_movie_id = {idx: movie_id for movie_id, idx in self.item_mapping.items()}
        self.tmdb_to_item = self._build_tmdb_index(self.catalog)

    @classmethod
    def from_dir(cls, artifacts_dir: str | Path) -> "HybridArtifactRecommender":
        return cls(load_artifact_bundle(artifacts_dir))

    @staticmethod
    def _build_tmdb_index(catalog: pd.DataFrame) -> dict[str, int]:
        result: dict[str, int] = {}
        if "tmdb_id" not in catalog.columns:
            return result
        for idx, value in enumerate(catalog["tmdb_id"].tolist()):
            if pd.isna(value):
                continue
            result[str(int(value))] = idx
            result[f"tmdb_{int(value)}"] = idx
        return result

    def search_movies(self, query: str = "", limit: int = 20) -> list[dict]:
        query = query.strip()
        catalog = self.catalog
        if query:
            pattern = re.escape(query)
            mask = catalog["title"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            catalog = catalog.loc[mask]
        return [
            self._movie_payload(int(row.item_idx), score=None, explanation_tags=[])
            for row in catalog.head(limit).itertuples(index=False)
        ]

    def _context_to_item_indices(self, session_context: Iterable[str | int] | None) -> list[int]:
        if not session_context:
            return []
        indices: list[int] = []
        title_to_idx = {str(row.title).lower(): int(row.item_idx) for row in self.catalog.itertuples(index=False)}
        for value in session_context:
            text = str(value).strip()
            if text in self.tmdb_to_item:
                indices.append(self.tmdb_to_item[text])
                continue
            if text.startswith("ml_"):
                text = text[3:]
            if text.isdigit() and int(text) in self.item_mapping:
                indices.append(self.item_mapping[int(text)])
                continue
            if text.lower() in title_to_idx:
                indices.append(title_to_idx[text.lower()])
        return sorted(set(indices))

    def _user_profile(self, user_id: int | None, session_item_indices: list[int]) -> np.ndarray:
        profiles = []
        if user_id is not None and int(user_id) in self.user_mapping:
            profiles.append(self.bundle.user_profiles[self.user_mapping[int(user_id)]])
        if session_item_indices:
            profiles.append(self.bundle.content_embeddings[session_item_indices].mean(axis=0))
        if not profiles:
            popularity_profile = np.average(
                self.bundle.content_embeddings,
                axis=0,
                weights=np.maximum(self.bundle.item_popularity, 1e-6),
            )
            profiles.append(popularity_profile)
        profile = np.mean(np.vstack(profiles), axis=0, keepdims=True)
        return normalize(profile).astype(np.float32)[0]

    def recommend(
        self,
        user_id: int | None = None,
        top_k: int = 10,
        session_context: Iterable[str | int] | None = None,
    ) -> list[dict]:
        top_k = max(1, min(int(top_k), 100))
        session_item_indices = self._context_to_item_indices(session_context)
        profile = self._user_profile(user_id, session_item_indices)

        content_scores = profile @ self.bundle.content_embeddings.T
        popularity_scores = self.bundle.item_popularity.astype(np.float32)
        cf_scores = np.zeros_like(content_scores, dtype=np.float32)
        has_cf = (
            user_id is not None
            and int(user_id) in self.user_mapping
            and self.bundle.lightgcn_user_embeddings is not None
            and self.bundle.lightgcn_item_embeddings is not None
        )
        if has_cf:
            user_idx = self.user_mapping[int(user_id)]
            cf_scores = self.bundle.lightgcn_user_embeddings[user_idx] @ self.bundle.lightgcn_item_embeddings.T

        weights = self.bundle.hybrid_config or {}
        cf_weight = float(weights.get("cf_weight", 0.45 if has_cf else 0.0))
        content_weight = float(weights.get("content_weight", 0.45 if has_cf else 0.85))
        popularity_weight = float(weights.get("popularity_weight", 0.10 if has_cf else 0.15))

        scores = (
            cf_weight * minmax(cf_scores)
            + content_weight * minmax(content_scores)
            + popularity_weight * minmax(popularity_scores)
        )

        blocked = set(session_item_indices)
        if user_id is not None and int(user_id) in self.user_mapping:
            watched = self.bundle.hybrid_config.get("train_user_items", {}).get(str(self.user_mapping[int(user_id)]), [])
            blocked.update(int(item) for item in watched)
        if blocked:
            scores[list(blocked)] = -np.inf

        top_indices = top_k_from_scores(scores, top_k)
        return [self._movie_payload(idx, float(scores[idx]), self._explanations(idx, session_item_indices, has_cf)) for idx in top_indices]

    def _movie_payload(self, item_idx: int, score: float | None, explanation_tags: list[str]) -> dict:
        row = self.catalog.iloc[item_idx]
        return {
            "movie_id": int(row.get("movieId", self.index_to_movie_id.get(item_idx, item_idx))),
            "tmdb_id": None if "tmdb_id" not in row or pd.isna(row.get("tmdb_id")) else int(row.get("tmdb_id")),
            "title": str(row.get("title", "")),
            "score": score,
            "poster_url": row.get("poster_url") if pd.notna(row.get("poster_url", None)) else None,
            "explanation_tags": explanation_tags,
        }

    def _explanations(self, item_idx: int, session_item_indices: list[int], has_cf: bool) -> list[str]:
        tags: list[str] = []
        row = self.catalog.iloc[item_idx]
        if has_cf:
            tags.append("phù hợp lịch sử đánh giá")
        if session_item_indices:
            session_genres = set()
            for idx in session_item_indices:
                session_genres.update(str(self.catalog.iloc[idx].get("genres", "")).split("|"))
                session_genres.update(str(self.catalog.iloc[idx].get("tmdb_genres", "")).split("|"))
            row_genres = set(str(row.get("genres", "")).split("|")) | set(str(row.get("tmdb_genres", "")).split("|"))
            overlap = sorted(g for g in session_genres & row_genres if g and g != "(no genres listed)")
            if overlap:
                tags.append("cùng thể loại: " + ", ".join(overlap[:3]))
        if row.get("director"):
            tags.append(f"đạo diễn {row.get('director')}")
        if not tags:
            tags.append("phổ biến trong tập dữ liệu")
        return tags[:3]
