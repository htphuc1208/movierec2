from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        collaborative_factors: int = 24,
        collaborative_epochs: int = 30,
        collaborative_lr: float = 0.01,
        collaborative_reg: float = 0.02,
        collaborative_batch_size: int = 2048,
        collaborative_engine: str = "auto",
        collaborative_optimizer: str = "adamw",
        collaborative_init_std: float = 0.05,
        collaborative_bias_reg: float = 0.005,
        collaborative_bias_shrinkage: float = 5.0,
        collaborative_momentum: float = 0.0,
        collaborative_max_grad_norm: float = 5.0,
        collaborative_validation_ratio: float = 0.0,
        collaborative_patience: int = 6,
        content_backend: str = "tfidf",
        content_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
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
        self.collaborative_engine = collaborative_engine
        self.collaborative_optimizer = collaborative_optimizer
        self.collaborative_init_std = collaborative_init_std
        self.collaborative_bias_reg = collaborative_bias_reg
        self.collaborative_bias_shrinkage = collaborative_bias_shrinkage
        self.collaborative_momentum = collaborative_momentum
        self.collaborative_max_grad_norm = collaborative_max_grad_norm
        self.collaborative_validation_ratio = collaborative_validation_ratio
        self.collaborative_patience = collaborative_patience
        self.content_backend = content_backend
        self.content_model_name = content_model_name
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
        self._content_backend_used = "unfit"
        self._content_from_artifact = False
        self._popularity: np.ndarray | None = None
        self._collaborative_mode = "funk_svd"
        self._collaborative_engine_used = "unfit"
        self.model_source = "unfit"
        self.model_name = "hybrid-pytorch-svd-tfidf"
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
        self.model_name = (
            "hybrid-pytorch-svd-tfidf"
            if self._collaborative_engine_used == "torch"
            else "hybrid-funk-svd-tfidf"
        )
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
        content_manifest = manifest.get("content", {}) if isinstance(manifest.get("content", {}), dict) else {}
        requested_content_backend = self.content_backend
        manifest_content_backend = str(content_manifest.get("backend", "")).strip().lower()
        if manifest_content_backend and manifest_content_backend != "tfidf":
            self.content_backend = manifest_content_backend
        else:
            self.content_backend = requested_content_backend
        self.content_model_name = str(content_manifest.get("model_name", self.content_model_name))
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
        self._collaborative_engine_used = str(manifest.get("collaborative", {}).get("engine", "artifact"))

        if not self._load_content_artifact(artifact_path, manifest):
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
                "engine": self._collaborative_engine_used,
                "factors": int(self._user_factors.shape[1]) if self._user_factors is not None else 0,
            },
            "files": {"collaborative": "collaborative.npz"},
            "metrics": metrics or self.metrics,
        }
        content_manifest = self._export_content_artifact(path)
        manifest["content"] = content_manifest
        if content_manifest.get("file"):
            manifest["files"]["content"] = content_manifest["file"]
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
            "collaborative_engine": self._collaborative_engine_used,
            "content_backend": self._content_backend_used,
            "content_model_name": self.content_model_name if self._content_backend_used == "sbert" else "",
            "content_from_artifact": self._content_from_artifact,
            "metrics": self.metrics,
            "user_count": len(self.user_ids),
            "movie_count": len(self.movie_ids),
        }

    def _build_collaborative_space(self) -> None:
        engine = self.collaborative_engine.lower().strip()
        if engine not in {"auto", "torch", "numpy"}:
            raise ValueError("collaborative_engine must be one of: auto, torch, numpy")

        if engine in {"auto", "torch"}:
            try:
                if self._build_torch_collaborative_space():
                    return
                if engine == "torch":
                    raise ImportError("Torch is required for collaborative_engine='torch'.")
            except ImportError:
                if engine == "torch":
                    raise

        self._build_numpy_collaborative_space()

    def _rating_interactions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        interactions = []
        for row in self.ratings.itertuples():
            user_idx = self.user_index.get(int(row.userId))
            item_idx = self.movie_index.get(int(row.movieId))
            if user_idx is not None and item_idx is not None:
                interactions.append((user_idx, item_idx, float(row.rating)))
        if not interactions:
            empty_indices = np.asarray([], dtype=np.int64)
            empty_ratings = np.asarray([], dtype=np.float32)
            return empty_indices, empty_indices, empty_ratings
        data = np.asarray(interactions, dtype=np.float32)
        return data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)

    def _build_bias_priors(self, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
        shrink = max(float(shrinkage), 0.0)
        user_bias = np.zeros(len(self.user_ids), dtype=np.float32)
        item_bias = np.zeros(len(self.movie_ids), dtype=np.float32)

        user_stats = self.ratings.groupby("userId")["rating"].agg(["mean", "count"])
        for user_id, row in user_stats.iterrows():
            idx = self.user_index.get(int(user_id))
            if idx is None:
                continue
            count = float(row["count"])
            user_bias[idx] = float(row["mean"] - self._global_mean) * count / (count + shrink)

        item_stats = self.ratings.groupby("movieId")["rating"].agg(["mean", "count"])
        for movie_id, row in item_stats.iterrows():
            idx = self.movie_index.get(int(movie_id))
            if idx is None:
                continue
            count = float(row["count"])
            item_bias[idx] = float(row["mean"] - self._global_mean) * count / (count + shrink)

        return user_bias, item_bias

    def _build_torch_collaborative_space(self) -> bool:
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from .SVD import SVDModel
        except ImportError:
            return False

        user_count = len(self.user_ids)
        item_count = len(self.movie_ids)
        factor_count = max(1, min(self.collaborative_factors, max(user_count, 1), max(item_count, 1)))
        self._global_mean = float(self.ratings["rating"].mean()) if not self.ratings.empty else 3.0
        self._user_factors = np.zeros((user_count, factor_count), dtype=np.float32)
        self._item_factors = np.zeros((item_count, factor_count), dtype=np.float32)
        self._user_bias = np.zeros(user_count, dtype=np.float32)
        self._item_bias = np.zeros(item_count, dtype=np.float32)

        user_indices, item_indices, rating_values = self._rating_interactions()
        self._rating_matrix = np.zeros((user_count, item_count), dtype=np.float32)
        if len(rating_values):
            self._rating_matrix[user_indices, item_indices] = rating_values
        if user_count == 0 or item_count == 0 or len(rating_values) == 0:
            self._collaborative_engine_used = "torch"
            return True

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SVDModel(
            num_users=user_count,
            num_items=item_count,
            embedding_dim=factor_count,
            global_mean=self._global_mean,
            init_std=self.collaborative_init_std,
        ).to(device)

        if self.collaborative_bias_shrinkage >= 0:
            user_bias, item_bias = self._build_bias_priors(self.collaborative_bias_shrinkage)
            model.initialize_biases(torch.tensor(user_bias), torch.tensor(item_bias))

        train_positions, val_positions = self._train_validation_positions(user_indices)
        train_users = torch.tensor(user_indices[train_positions], dtype=torch.long)
        train_items = torch.tensor(item_indices[train_positions], dtype=torch.long)
        train_labels = torch.tensor(rating_values[train_positions], dtype=torch.float32)
        dataset = TensorDataset(train_users, train_items, train_labels)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.collaborative_batch_size),
            shuffle=True,
            generator=generator,
        )

        optimizer_name = self.collaborative_optimizer.lower().strip()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=self.collaborative_lr, momentum=self.collaborative_momentum)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.collaborative_lr)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.collaborative_lr)

        criterion = torch.nn.MSELoss()
        best_loss = float("inf")
        best_state = None
        patience_left = max(1, self.collaborative_patience)
        val_users = torch.tensor(user_indices[val_positions], dtype=torch.long, device=device) if len(val_positions) else None
        val_items = torch.tensor(item_indices[val_positions], dtype=torch.long, device=device) if len(val_positions) else None
        val_labels = torch.tensor(rating_values[val_positions], dtype=torch.float32, device=device) if len(val_positions) else None

        for _ in range(max(0, self.collaborative_epochs)):
            model.train()
            for batch_users, batch_items, batch_labels in loader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                preds = model(batch_users, batch_items)
                loss = criterion(preds, batch_labels)
                if self.collaborative_reg > 0:
                    user_vecs = model.user_embedding(batch_users)
                    item_vecs = model.item_embedding(batch_items)
                    loss = loss + self.collaborative_reg * (
                        user_vecs.pow(2).sum(dim=1) + item_vecs.pow(2).sum(dim=1)
                    ).mean()
                if self.collaborative_bias_reg > 0:
                    user_bias = model.user_bias(batch_users).squeeze(1)
                    item_bias = model.item_bias(batch_items).squeeze(1)
                    loss = loss + self.collaborative_bias_reg * (user_bias.pow(2) + item_bias.pow(2)).mean()
                loss.backward()
                if self.collaborative_max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.collaborative_max_grad_norm)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                if val_users is not None and val_items is not None and val_labels is not None and len(val_positions):
                    val_preds = model(val_users, val_items).clamp(0.5, 5.0)
                    epoch_loss = float(torch.mean((val_preds - val_labels) ** 2).item())
                else:
                    train_preds = model(train_users.to(device), train_items.to(device)).clamp(0.5, 5.0)
                    epoch_loss = float(torch.mean((train_preds - train_labels.to(device)) ** 2).item())

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = max(1, self.collaborative_patience)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            self._user_factors = model.user_embedding.weight.detach().cpu().numpy().astype(np.float32)
            self._item_factors = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
            self._user_bias = model.user_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
            self._item_bias = model.item_bias.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
            self._global_mean = float(model.global_mean.detach().cpu().item())
        self._collaborative_engine_used = "torch"
        return True

    def _train_validation_positions(self, user_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = len(user_indices)
        if total == 0 or self.collaborative_validation_ratio <= 0:
            positions = np.arange(total, dtype=np.int64)
            return positions, np.asarray([], dtype=np.int64)

        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for user_idx in np.unique(user_indices):
            positions = np.flatnonzero(user_indices == user_idx)
            if len(positions) < 5:
                train_parts.append(positions)
                continue
            val_count = max(1, int(round(len(positions) * self.collaborative_validation_ratio)))
            val_count = min(val_count, len(positions) - 1)
            train_parts.append(positions[:-val_count])
            val_parts.append(positions[-val_count:])

        train_positions = np.concatenate(train_parts).astype(np.int64) if train_parts else np.arange(total, dtype=np.int64)
        val_positions = np.concatenate(val_parts).astype(np.int64) if val_parts else np.asarray([], dtype=np.int64)
        if len(train_positions) == 0:
            train_positions = np.arange(total, dtype=np.int64)
            val_positions = np.asarray([], dtype=np.int64)
        return train_positions, val_positions

    def _build_numpy_collaborative_space(self) -> None:
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
            self._collaborative_engine_used = "numpy"
            return

        user_indices, item_indices, ratings = self._rating_interactions()
        if not len(ratings):
            self._rating_matrix = np.zeros((user_count, item_count), dtype=np.float32)
            self._collaborative_engine_used = "numpy"
            return

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
                    -self.collaborative_lr * (errors + self.collaborative_bias_reg * user_bias),
                )
                np.add.at(
                    self._item_bias,
                    items,
                    -self.collaborative_lr * (errors + self.collaborative_bias_reg * item_bias),
                )
        self._collaborative_engine_used = "numpy"

    def _build_content_space(self) -> None:
        if self._content_from_artifact and self._content_matrix is not None:
            return

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
                str(getattr(row, "keywords", "")).replace("|", " "),
                str(getattr(row, "tag_genome_tags", "")).replace("|", " "),
                str(getattr(row, "original_language", "")),
                str(getattr(row, "production_companies", "")).replace("|", " "),
                str(getattr(row, "production_countries", "")).replace("|", " "),
                str(getattr(row, "collection_name", "")),
                str(getattr(row, "certification", "")),
                " ".join(tag_map.get(movie_id, [])),
            ]
            text.append(" ".join(parts))

        backend = str(self.content_backend or "tfidf").lower().strip()
        if backend not in {"auto", "tfidf", "sbert"}:
            raise ValueError("content_backend must be one of: auto, tfidf, sbert")

        if backend in {"auto", "sbert"}:
            try:
                from .TwoTower import MetadataEncoder

                encoder = MetadataEncoder(model_name=self.content_model_name, backend="sbert")
                encoded = encoder.fit_transform(self.movies, self.tags)
                self._content_matrix = encoded.vectors.astype(np.float32)
                self._vectorizer = None
                self._content_backend_used = "sbert"
                self._content_from_artifact = False
                return
            except Exception:
                if backend == "sbert":
                    raise

        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.85)
        try:
            self._content_matrix = self._vectorizer.fit_transform(text)
        except ValueError:
            # Fallback for very small catalogs where min_df=2 prunes all terms
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            self._content_matrix = self._vectorizer.fit_transform(text)
        self._content_backend_used = "tfidf"
        self._content_from_artifact = False

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
        return self._global_mean_collaborative_scores()

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
            return np.zeros(len(self.movie_ids), dtype=np.float32)

        profile_vector = np.asarray(self._content_matrix[profile_indices].mean(axis=0)).reshape(1, -1)
        scores = cosine_similarity(profile_vector, self._content_matrix).ravel()
        return scores.astype(np.float32)

    def _popularity_scores(self) -> np.ndarray:
        self._ensure_fit()
        return self._popularity.astype(np.float32)

    def _global_mean_collaborative_scores(self) -> np.ndarray:
        """Fallback collaborative scores for unknown users using global-mean profile."""
        self._ensure_fit()
        if self._collaborative_mode == "embedding":
            mean_user = np.mean(self._user_factors, axis=0)
            scores = mean_user @ self._item_factors.T
            return scores.astype(np.float32)
        mean_user_bias = float(np.mean(self._user_bias)) if self._user_bias is not None else 0.0
        mean_user_factors = np.mean(self._user_factors, axis=0)
        scores = (
            self._global_mean
            + mean_user_bias
            + self._item_bias
            + mean_user_factors @ self._item_factors.T
        )
        return scores.astype(np.float32)

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

    def _load_content_artifact(self, artifact_path: Path, manifest: dict[str, Any]) -> bool:
        files = manifest.get("files", {}) if isinstance(manifest.get("files", {}), dict) else {}
        content_manifest = manifest.get("content", {}) if isinstance(manifest.get("content", {}), dict) else {}
        content_file = str(files.get("content") or content_manifest.get("file") or "").strip()
        if not content_file:
            return False
        artifact_backend = str(content_manifest.get("backend", "")).strip().lower()
        requested_backend = str(self.content_backend or "tfidf").strip().lower()
        if requested_backend in {"sbert", "auto"} and artifact_backend == "tfidf":
            return False

        content_path = artifact_path / content_file
        if not content_path.exists():
            return False

        data = np.load(content_path, allow_pickle=False)
        if "movie_ids" not in data or "item_vectors" not in data:
            return False

        artifact_movie_ids = data["movie_ids"].astype(np.int64).tolist()
        item_vectors = data["item_vectors"].astype(np.float32)
        self._content_matrix = self._align_item_embeddings(artifact_movie_ids, item_vectors)
        self._vectorizer = None
        self._content_backend_used = str(content_manifest.get("backend", "artifact"))
        self._content_from_artifact = True
        return True

    def _export_content_artifact(self, path: Path) -> dict[str, Any]:
        backend = self._content_backend_used if self._content_backend_used != "unfit" else "tfidf"
        manifest: dict[str, Any] = {"backend": backend}
        if backend == "sbert":
            manifest["model_name"] = self.content_model_name

        if self._content_matrix is None or sparse.issparse(self._content_matrix):
            return manifest

        item_vectors = np.asarray(self._content_matrix, dtype=np.float32)
        if item_vectors.ndim != 2 or item_vectors.shape[0] != len(self.movie_ids):
            return manifest

        np.savez_compressed(
            path / "content.npz",
            movie_ids=np.asarray(self.movie_ids, dtype=np.int64),
            item_vectors=item_vectors,
        )
        manifest["file"] = "content.npz"
        manifest["factors"] = int(item_vectors.shape[1])
        return manifest

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
        safe = scores.copy().astype(np.float64)
        safe[~finite] = np.nanmin(safe[finite]) - 1.0
        if np.nanmax(safe) == np.nanmin(safe):
            return np.zeros_like(safe, dtype=np.float32)
        ranked = rankdata(safe, method="average")
        scaled = (ranked - 1.0) / max(len(ranked) - 1.0, 1.0)
        return scaled.astype(np.float32)

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
        if self.movies is None or self.ratings is None or self._popularity is None or self._content_matrix is None:
            raise RuntimeError("HybridMovieRecommender.fit must be called before inference.")
