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
from recommender.inference.ratings_store import SidecarRatingStore


class HybridArtifactRecommender:
    """Serve recommendations and catalog views from exported artifacts."""

    def __init__(self, bundle: ArtifactBundle, artifacts_dir: str | Path | None = None) -> None:
        self.bundle = bundle
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else None
        self.catalog = bundle.catalog.reset_index(drop=True).copy()
        self.catalog["item_idx"] = np.arange(len(self.catalog))
        self.item_mapping = bundle.item_mapping
        self.user_mapping = bundle.user_mapping
        self.index_to_movie_id = {idx: movie_id for movie_id, idx in self.item_mapping.items()}
        self.tmdb_to_item = self._build_tmdb_index(self.catalog)
        self._title_to_item = {str(row.title).lower(): int(row.item_idx) for row in self.catalog.itertuples(index=False)}
        self.strong_ranker = None
        self.strong_ranker_error = ""
        self._load_strong_ranker()

    @classmethod
    def from_dir(cls, artifacts_dir: str | Path) -> "HybridArtifactRecommender":
        return cls(load_artifact_bundle(artifacts_dir), artifacts_dir=artifacts_dir)

    @staticmethod
    def _build_tmdb_index(catalog: pd.DataFrame) -> dict[str, int]:
        result: dict[str, int] = {}
        if "tmdb_id" not in catalog.columns:
            return result
        for idx, value in enumerate(catalog["tmdb_id"].tolist()):
            if pd.isna(value):
                continue
            try:
                tmdb_id = int(value)
            except (TypeError, ValueError):
                continue
            result[str(tmdb_id)] = idx
            result[f"tmdb_{tmdb_id}"] = idx
        return result

    def users(self) -> list[int]:
        return sorted(int(user_id) for user_id in self.user_mapping)

    def model_info(self) -> dict:
        metrics = self.bundle.metrics or {}
        config = self.bundle.hybrid_config or {}
        return {
            "model_source": "artifact",
            "model_name": config.get("model_type", "hybrid-artifact-lightgcn-content"),
            "user_count": len(self.user_mapping),
            "movie_count": len(self.item_mapping),
            "has_lightgcn": self.bundle.lightgcn_user_embeddings is not None and self.bundle.lightgcn_item_embeddings is not None,
            "has_two_tower": self.bundle.two_tower_user_embeddings is not None and self.bundle.two_tower_item_embeddings is not None,
            "has_strong_ranker": self.strong_ranker is not None,
            "strong_ranker_error": self.strong_ranker_error,
            "weights": {
                "cf": float(config.get("cf_weight", 0.0)),
                "two_tower": float(config.get("two_tower_weight", 0.0)),
                "content": float(config.get("content_weight", 0.0)),
                "popularity": float(config.get("popularity_weight", 0.0)),
            },
            "content_backend": config.get("content_backend", ""),
            "metrics": metrics,
        }

    def search_movies(self, query: str = "", limit: int = 20) -> list[dict]:
        query = query.strip()
        catalog = self.catalog
        if query:
            pattern = re.escape(query)
            mask = catalog["title"].astype(str).str.contains(pattern, case=False, na=False, regex=True)
            catalog = catalog.loc[mask]
        return [self._movie_payload(int(row.item_idx)) for row in catalog.head(limit).itertuples(index=False)]

    def movie_detail(self, movie_id: int | str) -> dict | None:
        item_idx = self._movie_id_to_item_idx(movie_id)
        if item_idx is None:
            return None
        return self._movie_payload(item_idx, include_detail=True)

    def similar_movies(self, movie_id: int | str, top_k: int = 15) -> list[dict]:
        item_idx = self._movie_id_to_item_idx(movie_id)
        if item_idx is None:
            return []
        embeddings = normalize(self.bundle.content_embeddings.astype(np.float32))
        scores = embeddings[item_idx] @ embeddings.T
        scores[item_idx] = -np.inf
        top_indices = top_k_from_scores(scores, max(1, min(int(top_k), 100)))
        return [
            self._movie_payload(idx, score=float(scores[idx]), explanation_tags=self._similarity_explanations(item_idx, idx))
            for idx in top_indices
            if np.isfinite(scores[idx])
        ]

    def trending_movies(self, top_k: int = 15) -> list[dict]:
        return self._rank_catalog(
            top_k,
            key=lambda item_idx: (
                self._numeric(item_idx, "vote_count"),
                self._numeric(item_idx, "popularity"),
                self._numeric(item_idx, "vote_average"),
            ),
        )

    def top_rated_movies(self, top_k: int = 15) -> list[dict]:
        return self._rank_catalog(
            top_k,
            key=lambda item_idx: (
                self._numeric(item_idx, "vote_average"),
                self._numeric(item_idx, "vote_count"),
                self._numeric(item_idx, "popularity"),
            ),
        )

    def latest_movies(self, top_k: int = 15) -> list[dict]:
        return self._rank_catalog(
            top_k,
            key=lambda item_idx: (
                self._release_year(item_idx),
                self._numeric(item_idx, "popularity"),
                self._numeric(item_idx, "vote_count"),
            ),
        )

    def genre_movies(self, genre: str, top_k: int = 15) -> list[dict]:
        needle = genre.strip().lower()
        if not needle:
            return []
        candidates = []
        for row in self.catalog.itertuples(index=False):
            text = f"{getattr(row, 'genres', '')}|{getattr(row, 'tmdb_genres', '')}".lower()
            if needle in text:
                candidates.append(int(row.item_idx))
        candidates.sort(
            key=lambda item_idx: (
                self._numeric(item_idx, "vote_average"),
                self._numeric(item_idx, "vote_count"),
                self._numeric(item_idx, "popularity"),
            ),
            reverse=True,
        )
        return [self._movie_payload(idx) for idx in candidates[: max(1, min(int(top_k), 100))]]

    def user_history(self, user_id: int, rating_store: SidecarRatingStore | None = None, top_k: int = 15) -> list[dict]:
        top_k = max(1, min(int(top_k), 100))
        seen: set[int] = set()
        results: list[dict] = []

        if rating_store is not None:
            for rating in rating_store.ratings_for_user(int(user_id)):
                item_idx = self._movie_id_to_item_idx(rating["movie_id"])
                if item_idx is None:
                    continue
                payload = self._movie_payload(item_idx)
                payload["user_rating"] = float(rating["rating"])
                payload["rating_timestamp"] = int(rating["timestamp"])
                payload["history_source"] = "sidecar"
                results.append(payload)
                seen.add(item_idx)
                if len(results) >= top_k:
                    return results

        user_idx = self.user_mapping.get(int(user_id))
        if user_idx is None:
            return results
        train_items = self.bundle.hybrid_config.get("train_user_items", {}).get(str(user_idx), [])
        train_item_indices = [int(item) for item in train_items if 0 <= int(item) < len(self.catalog) and int(item) not in seen]
        train_item_indices.sort(key=lambda item_idx: float(self.bundle.item_popularity[item_idx]), reverse=True)
        for item_idx in train_item_indices:
            payload = self._movie_payload(item_idx)
            payload["user_rating"] = None
            payload["history_source"] = "train"
            results.append(payload)
            if len(results) >= top_k:
                break
        return results

    def recommend(
        self,
        user_id: int | None = None,
        top_k: int = 10,
        session_context: Iterable[str | int] | None = None,
        exclude_seen: bool = True,
        model_name: str = "hybrid",
    ) -> list[dict]:
        top_k = max(1, min(int(top_k), 100))
        session_item_indices = self._context_to_item_indices(session_context)
        content_scores = self._content_scores(user_id, session_item_indices)
        popularity_scores = self.bundle.item_popularity.astype(np.float32)
        cf_scores, has_cf = self._cf_scores(user_id)
        two_tower_scores, has_two_tower = self._two_tower_scores(user_id)
        strong_scores, has_strong = self._strong_ranker_scores(user_id, allow_session=not session_item_indices)
        mode = self._normalise_model_name(model_name)

        if mode == "strong" and has_strong:
            scores = minmax(strong_scores)
        elif mode == "hybrid" and has_strong and self.bundle.hybrid_config.get("model_type") == "strong_ranker":
            scores = minmax(strong_scores)
        elif mode == "lightgcn" and has_cf:
            scores = minmax(cf_scores)
        elif mode == "two_tower" and has_two_tower:
            scores = minmax(two_tower_scores)
        elif mode == "content":
            scores = minmax(content_scores)
        elif mode == "popularity":
            scores = minmax(popularity_scores)
        else:
            scores = self._hybrid_scores(cf_scores, two_tower_scores, content_scores, popularity_scores, has_cf, has_two_tower)

        blocked = set(session_item_indices)
        if exclude_seen and user_id is not None and int(user_id) in self.user_mapping:
            user_idx = self.user_mapping[int(user_id)]
            watched = self.bundle.hybrid_config.get("train_user_items", {}).get(str(user_idx), [])
            blocked.update(int(item) for item in watched)
        if blocked:
            scores[list(blocked)] = -np.inf

        top_indices = top_k_from_scores(scores, top_k)
        return [
            self._movie_payload(
                idx,
                score=float(scores[idx]),
                explanation_tags=self._explanations(idx, session_item_indices, has_cf=(mode == "lightgcn" and has_cf) or has_cf),
                match_score=self._match_score(scores[idx]),
            )
            for idx in top_indices
            if np.isfinite(scores[idx])
        ]

    def _content_scores(self, user_id: int | None, session_item_indices: list[int]) -> np.ndarray:
        profile = self._user_profile(user_id, session_item_indices)
        return (profile @ self.bundle.content_embeddings.T).astype(np.float32)

    def _cf_scores(self, user_id: int | None) -> tuple[np.ndarray, bool]:
        scores = np.zeros(len(self.catalog), dtype=np.float32)
        has_cf = (
            user_id is not None
            and int(user_id) in self.user_mapping
            and self.bundle.lightgcn_user_embeddings is not None
            and self.bundle.lightgcn_item_embeddings is not None
        )
        if has_cf:
            user_idx = self.user_mapping[int(user_id)]
            scores = (self.bundle.lightgcn_user_embeddings[user_idx] @ self.bundle.lightgcn_item_embeddings.T).astype(np.float32)
        return scores, has_cf

    def _two_tower_scores(self, user_id: int | None) -> tuple[np.ndarray, bool]:
        scores = np.zeros(len(self.catalog), dtype=np.float32)
        has_two_tower = (
            user_id is not None
            and int(user_id) in self.user_mapping
            and self.bundle.two_tower_user_embeddings is not None
            and self.bundle.two_tower_item_embeddings is not None
        )
        if has_two_tower:
            user_idx = self.user_mapping[int(user_id)]
            scores = (self.bundle.two_tower_user_embeddings[user_idx] @ self.bundle.two_tower_item_embeddings.T).astype(np.float32)
        return scores, has_two_tower

    def _strong_ranker_scores(self, user_id: int | None, allow_session: bool = True) -> tuple[np.ndarray, bool]:
        scores = np.zeros(len(self.catalog), dtype=np.float32)
        if not allow_session or self.strong_ranker is None or user_id is None or int(user_id) not in self.user_mapping:
            return scores, False
        user_idx = self.user_mapping[int(user_id)]
        try:
            ranker_scores = self.strong_ranker.score_users(np.asarray([user_idx], dtype=np.int64))[0]
        except Exception as exc:
            self.strong_ranker_error = str(exc)
            return scores, False
        if len(ranker_scores) != len(self.catalog):
            self.strong_ranker_error = f"ranker returned {len(ranker_scores)} scores for {len(self.catalog)} catalog items"
            return scores, False
        return np.asarray(ranker_scores, dtype=np.float32), True

    def _hybrid_scores(
        self,
        cf_scores: np.ndarray,
        two_tower_scores: np.ndarray,
        content_scores: np.ndarray,
        popularity_scores: np.ndarray,
        has_cf: bool,
        has_two_tower: bool,
    ) -> np.ndarray:
        weights = self.bundle.hybrid_config or {}
        cf_weight = float(weights.get("cf_weight", 0.45 if has_cf else 0.0))
        two_tower_weight = float(weights.get("two_tower_weight", 0.0 if not has_two_tower else 0.35))
        content_weight = float(weights.get("content_weight", 0.45 if has_cf else 0.85))
        popularity_weight = float(weights.get("popularity_weight", 0.10 if has_cf else 0.15))
        return (
            cf_weight * minmax(cf_scores)
            + two_tower_weight * minmax(two_tower_scores)
            + content_weight * minmax(content_scores)
            + popularity_weight * minmax(popularity_scores)
        ).astype(np.float32)

    def _context_to_item_indices(self, session_context: Iterable[str | int] | None) -> list[int]:
        if not session_context:
            return []
        indices: list[int] = []
        for value in session_context:
            text = str(value).strip()
            if text in self.tmdb_to_item:
                indices.append(self.tmdb_to_item[text])
                continue
            if text.startswith(("ml_", "movie_")):
                text = text.split("_", 1)[1]
            if text.isdigit() and int(text) in self.item_mapping:
                indices.append(self.item_mapping[int(text)])
                continue
            title_idx = self._title_to_item.get(text.lower())
            if title_idx is not None:
                indices.append(title_idx)
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

    def _movie_payload(
        self,
        item_idx: int,
        score: float | None = None,
        explanation_tags: list[str] | None = None,
        include_detail: bool = False,
        match_score: float | None = None,
    ) -> dict:
        row = self.catalog.iloc[item_idx]
        movie_id = int(row.get("movieId", self.index_to_movie_id.get(item_idx, item_idx)))
        vote_average = self._json_scalar(row.get("vote_average", 0.0))
        payload = {
            "movie_id": movie_id,
            "movieId": movie_id,
            "tmdb_id": self._optional_int(row.get("tmdb_id")),
            "title": str(row.get("title", "")),
            "genres": str(row.get("genres", "")),
            "tmdb_genres": str(row.get("tmdb_genres", "")),
            "score": score if score is not None else float(vote_average or 0.0),
            "vote_average": float(vote_average or 0.0),
            "vote_count": int(float(self._json_scalar(row.get("vote_count", 0)) or 0)),
            "popularity": float(self._json_scalar(row.get("popularity", 0.0)) or 0.0),
            "poster_url": self._optional_str(row.get("poster_url")),
            "release_date": self._optional_str(row.get("release_date")),
            "release_year": self._release_year(item_idx),
            "year": self._release_year(item_idx),
            "overview": self._optional_str(row.get("overview")) or "Chưa có thông tin tóm tắt cho bộ phim này.",
            "director": self._optional_str(row.get("director")),
            "cast": self._optional_str(row.get("cast")),
            "runtime_minutes": int(float(self._json_scalar(row.get("runtime_minutes", 0)) or 0)),
            "explanation_tags": explanation_tags or [],
        }
        if match_score is not None:
            payload["match_score"] = match_score
        if include_detail:
            for field in ["keywords", "writers", "production_companies", "production_countries", "collection", "original_language"]:
                payload[field] = self._optional_str(row.get(field))
        return payload

    def _movie_id_to_item_idx(self, movie_id: int | str) -> int | None:
        text = str(movie_id).strip()
        if text.startswith("ml_"):
            text = text[3:]
        if not text.isdigit():
            return None
        return self.item_mapping.get(int(text))

    def _rank_catalog(self, top_k: int, key) -> list[dict]:
        item_indices = list(range(len(self.catalog)))
        item_indices.sort(key=key, reverse=True)
        return [self._movie_payload(idx) for idx in item_indices[: max(1, min(int(top_k), 100))]]

    def _numeric(self, item_idx: int, field: str) -> float:
        return float(self._json_scalar(self.catalog.iloc[item_idx].get(field, 0.0)) or 0.0)

    def _release_year(self, item_idx: int) -> int:
        row = self.catalog.iloc[item_idx]
        year = self._json_scalar(row.get("release_year", ""))
        if str(year).isdigit():
            return int(year)
        release_date = str(row.get("release_date", ""))
        return int(release_date[:4]) if release_date[:4].isdigit() else 0

    def _match_score(self, score: float) -> float:
        if not np.isfinite(score):
            return 0.0
        return float(max(0.0, min(1.0, score)))

    def _normalise_model_name(self, model_name: str | None) -> str:
        text = (model_name or "hybrid").strip().lower()
        if "lightgcn" in text:
            return "lightgcn"
        if "ranker" in text or "strong" in text or "lightgbm" in text:
            return "strong"
        if "two" in text and "tower" in text:
            return "two_tower"
        if "content" in text or "tfidf" in text or "sbert" in text:
            return "content"
        if "popular" in text:
            return "popularity"
        return "hybrid"

    def _load_strong_ranker(self) -> None:
        config = self.bundle.hybrid_config or {}
        ranker_path = config.get("ranker_path")
        if not self.artifacts_dir or not ranker_path:
            return
        path = self.artifacts_dir / str(ranker_path)
        if not path.exists():
            self.strong_ranker_error = f"{path.name} not found"
            return
        try:
            import joblib

            self.strong_ranker = joblib.load(path)
        except Exception as exc:
            self.strong_ranker_error = str(exc)

    def _similarity_explanations(self, source_idx: int, item_idx: int) -> list[str]:
        overlap = self._genre_overlap(source_idx, item_idx)
        if overlap:
            return ["cùng thể loại: " + ", ".join(overlap[:3])]
        return ["tương tự theo metadata"]

    def _explanations(self, item_idx: int, session_item_indices: list[int], has_cf: bool) -> list[str]:
        tags: list[str] = []
        row = self.catalog.iloc[item_idx]
        if has_cf:
            tags.append("phù hợp lịch sử đánh giá")
        if session_item_indices:
            overlap = sorted({genre for source_idx in session_item_indices for genre in self._genre_overlap(source_idx, item_idx)})
            if overlap:
                tags.append("cùng thể loại: " + ", ".join(overlap[:3]))
        director = self._optional_str(row.get("director"))
        if director:
            tags.append(f"đạo diễn {director.split('|')[0]}")
        if not tags:
            tags.append("phổ biến trong tập dữ liệu")
        return tags[:3]

    def _genre_overlap(self, left_idx: int, right_idx: int) -> list[str]:
        def genres(idx: int) -> set[str]:
            row = self.catalog.iloc[idx]
            values = f"{row.get('genres', '')}|{row.get('tmdb_genres', '')}".replace(",", "|").split("|")
            return {value.strip() for value in values if value.strip() and value.strip() != "(no genres listed)"}

        return sorted(genres(left_idx) & genres(right_idx))

    @staticmethod
    def _optional_int(value) -> int | None:
        if pd.isna(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_str(value) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    @staticmethod
    def _json_scalar(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
