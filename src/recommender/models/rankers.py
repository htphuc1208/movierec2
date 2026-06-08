"""Hybrid weighted and learned rankers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recommender.eval.metrics import evaluate_score_fn, minmax

# đây là một ranker đơn giản học cách kết hợp các component recommender 
# khác nhau bằng cách gán trọng số cho chúng, 
# sau đó sử dụng một ranker mạnh hơn (LightGBM hoặc SGDClassifier) 
# để học cách kết hợp các component này với nhau và với các đặc trưng của item từ catalog để đưa ra xếp hạng cuối cùng. 
# Mục tiêu là tận dụng sức mạnh của nhiều component khác nhau và học cách kết hợp chúng một cách hiệu quả dựa trên dữ liệu huấn luyện.
@dataclass
class WeightedHybridRecommender:
    components: list[Any]
    include_popularity: bool = True
    tune: bool = True
    k: int = 10
    grid_step: float = 0.25
    name: str = "hybrid_weighted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "WeightedHybridRecommender":
        self.dataset_ = dataset
        # tính độ phổ biến của item dựa trên ma trận train, chuẩn hóa về [0, 1]
        self.popularity_ = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        if self.popularity_.max() > 0:
            self.popularity_ = self.popularity_ / self.popularity_.max()
        component_count = len(self.components) + (1 if self.include_popularity else 0)
        if not self.tune or not dataset.val_user_items:
            self.weights_ = np.ones(component_count, dtype=np.float32) / max(1, component_count)
            self.metadata = {"weights": self.weights_.tolist(), "tuned": False}
            return self
        
        # nếu có tune trọng số:
        candidates = _weight_grid(component_count, step=self.grid_step)
        best_weights = candidates[0]
        best_ndcg = -1.0
        for weights in candidates:
            metrics = evaluate_score_fn(
                dataset.num_users,
                dataset.num_items,
                lambda users, w=weights: self._score_with_weights(users, w),
                dataset.train_user_items,
                dataset.val_user_items,
                k=self.k,
            )
            # ưu tiên ndcg@k để chọn trọng số tốt nhất, 
            # nếu có nhiều bộ trọng số có ndcg@k bằng nhau thì sẽ chọn bộ đầu tiên 
            ndcg = metrics.get(f"ndcg@{self.k}", 0.0)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_weights = weights
        self.weights_ = best_weights.astype(np.float32)
        self.metadata = {"weights": self.weights_.tolist(), "tuned": True, "grid_step": self.grid_step, f"validation_ndcg@{self.k}": best_ndcg}
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        return self._score_with_weights(user_indices, self.weights_)

    # scale bằng min-max để đưa điểm số của từng component về cùng thang đo, 
    # tránh việc một component có thang điểm lớn hơn sẽ chi phối kết quả cuối cùng, 
    # giúp học trọng số hiệu quả hơn
    def _score_with_weights(self, user_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(user_indices), self.dataset_.num_items), dtype=np.float32)
        offset = 0
        for component in self.components:
            scores += float(weights[offset]) * minmax(component.score_users(user_indices), axis=1)
            offset += 1
        if self.include_popularity:
            scores += float(weights[offset]) * minmax(np.broadcast_to(self.popularity_[None, :], scores.shape), axis=1)
        return scores

@dataclass
class SGDRankHybridRecommender:
    components: list[Any]
    include_popularity: bool = True
    negatives_per_positive: int = 2
    max_train_samples: int = 200_000
    seed: int = 42
    name: str = "hybrid_ranker"
    metadata: dict[str, Any] = field(default_factory=dict)
        
    # Với mỗi cặp (user, item), nó tạo feature:
    # score từ LightGCN
    # score từ TwoTower
    # score từ Content model
    # popularity của item
    # độ dài lịch sử user
    # Label là: 1 nếu user đã tương tác item, 0 nếu item negative sample
    # Model học feature của cặp user-item → xác suất user thích item

    def fit(self, dataset) -> "SGDRankHybridRecommender":
        self.dataset_ = dataset
        self.popularity_ = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        if self.popularity_.max() > 0:
            self.popularity_ = self.popularity_ / self.popularity_.max()
        users, items, labels = self._sample_pairs(dataset)
        features = self._features_for_pairs(users, items)
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=self.seed),
        )
        self.pipeline_.fit(features, labels)
        self.metadata = {
            "components": [component.name for component in self.components],
            "include_popularity": self.include_popularity,
            "train_samples": int(len(labels)),
        }
        return self
    # khi cần recommend, tạo feature cho tất cả cặp (i, u). sau đó dự đoán xác suất 
    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        all_items = np.arange(self.dataset_.num_items, dtype=np.int64)
        for user in user_indices:
            users = np.full(self.dataset_.num_items, int(user), dtype=np.int64)
            features = self._features_for_pairs(users, all_items)
            rows.append(self.pipeline_.predict_proba(features)[:, 1].astype(np.float32))
        return np.vstack(rows)

    def _sample_pairs(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positives = dataset.train[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        # với mỗi positive, sample một vài negative item:
        rng = np.random.default_rng(self.seed)
        if len(positives) > self.max_train_samples:
            positives = positives[rng.choice(len(positives), size=self.max_train_samples, replace=False)]
        neg_users: list[int] = []
        neg_items: list[int] = []
        for user, _ in positives:
            seen = dataset.train_user_items[int(user)]
            for _ in range(self.negatives_per_positive):
                item = _sample_negative_item(rng, dataset.num_items, seen)
                if item is None:
                    continue
                neg_users.append(int(user))
                neg_items.append(item)
        users = np.concatenate([positives[:, 0], np.asarray(neg_users, dtype=np.int64)])
        items = np.concatenate([positives[:, 1], np.asarray(neg_items, dtype=np.int64)])
        labels = np.concatenate([np.ones(len(positives), dtype=np.int64), np.zeros(len(neg_users), dtype=np.int64)])
        return users, items, labels
    # hàm này tạo feature cho từng cặp (user, item) 
    def _features_for_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        unique_users, inverse = np.unique(users, return_inverse=True)
        for component in self.components:
            score_matrix = component.score_users(unique_users)
            columns.append(score_matrix[inverse, items])
        if self.include_popularity:
            columns.append(self.popularity_[items])
        columns.append(np.asarray([len(self.dataset_.train_user_items.get(int(user), set())) for user in users], dtype=np.float32))
        return np.vstack(columns).T.astype(np.float32)


@dataclass
class StrongHybridRankerRecommender:
    """Learned ranker over recommender component scores and catalog features."""
    # ý tưởng là sử dụng một ranker mạnh hơn (LightGBM hoặc SGDClassifier) 
    # để học cách kết hợp các component recommender khác nhau với nhau 
    # và với các đặc trưng của item từ catalog để đưa ra xếp hạng cuối cùng.
    # Tức là nó không chỉ tin vào các model con, mà còn nhìn cả metadata phim.
    components: list[Any]
    include_popularity: bool = True
    negatives_per_positive: int = 4
    max_train_samples: int = 500_000
    seed: int = 42
    ranker: str = "auto"
    name: str = "hybrid_strong_ranker"
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, dataset) -> "StrongHybridRankerRecommender":
        self.dataset_ = dataset
        self.popularity_ = np.asarray(dataset.train_matrix.sum(axis=0)).ravel().astype(np.float32)
        if self.popularity_.max() > 0:
            self.popularity_ = self.popularity_ / self.popularity_.max()
        self.item_genres_ = [_genre_set(row) for _, row in dataset.catalog.iterrows()]
        # build user_genres_ để  gom tất cả thể loại phim mà user đã xem, 
        self.user_genres_ = self._build_user_genres(dataset)
        
        users, items, labels = self._sample_pairs(dataset)
        order = np.lexsort((items, users))
        users, items, labels = users[order], items[order], labels[order]
        features = self._features_for_pairs(users, items)
        # Thuật toán LGBMRanker cần biết các bộ phim nào thuộc về cùng một người dùng để gom chúng lại thành một nhóm (group).
        # nhóm theo user để sử dụng trong LightGBM Ranker, 
        # đảm bảo rằng các cặp (user, item) của cùng một user được đưa vào cùng một nhóm
        groups = _group_sizes(users)

        self.feature_names_ = self._feature_names()
        self.ranker_used_ = "sgd"
        try:
            if self.ranker in {"auto", "lightgbm"}:
                from lightgbm import LGBMRanker

                model = LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=self.seed,
                    verbose=-1,
                )
                model.fit(features, labels, group=groups)
                self.model_ = model
                self.ranker_used_ = "lightgbm"
            else:
                raise ImportError("LightGBM disabled")
        except Exception as exc:
            self.model_ = make_pipeline(
                StandardScaler(),
                SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=self.seed),
            )
            self.model_.fit(features, labels)
            self.ranker_used_ = "sgd_fallback"
            self.fallback_reason_ = str(exc)

        self.metadata = {
            "components": [component.name for component in self.components],
            "include_popularity": self.include_popularity,
            "train_samples": int(len(labels)),
            "ranker": self.ranker_used_,
            "feature_names": self.feature_names_,
        }
        if hasattr(self, "fallback_reason_"):
            self.metadata["fallback_reason"] = self.fallback_reason_
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        all_items = np.arange(self.dataset_.num_items, dtype=np.int64)
        for user in user_indices:
            users = np.full(self.dataset_.num_items, int(user), dtype=np.int64)
            features = self._features_for_pairs(users, all_items)
            if self.ranker_used_ == "lightgbm":
                predictions = self.model_.predict(features)
            else:
                predictions = self.model_.predict_proba(features)[:, 1]
            rows.append(np.asarray(predictions, dtype=np.float32))
        return np.vstack(rows)

    def _sample_pairs(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positive_frames = [dataset.train[["user_idx", "item_idx"]]]
        if not dataset.val.empty:
            positive_frames.append(dataset.val[["user_idx", "item_idx"]])
        positives = np.concatenate([frame.to_numpy(dtype=np.int64) for frame in positive_frames], axis=0)
        rng = np.random.default_rng(self.seed)
        if len(positives) > self.max_train_samples:
            positives = positives[rng.choice(len(positives), size=self.max_train_samples, replace=False)]
        known_user_items = {user: set(items) for user, items in dataset.train_user_items.items()}
        for user, items in dataset.val_user_items.items():
            known_user_items.setdefault(int(user), set()).update(items)
        neg_users: list[int] = []
        neg_items: list[int] = []
        for user, _ in positives:
            seen = known_user_items.get(int(user), set())
            for _ in range(self.negatives_per_positive):
                item = _sample_negative_item(rng, dataset.num_items, seen)
                if item is None:
                    continue
                neg_users.append(int(user))
                neg_items.append(item)
        users = np.concatenate([positives[:, 0], np.asarray(neg_users, dtype=np.int64)])
        items = np.concatenate([positives[:, 1], np.asarray(neg_items, dtype=np.int64)])
        labels = np.concatenate([np.ones(len(positives), dtype=np.int64), np.zeros(len(neg_users), dtype=np.int64)])
        return users, items, labels
    # với mỗi cặp (user, item), tạo feature dựa trên điểm số của các component recommender,
    def _features_for_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        unique_users, inverse = np.unique(users, return_inverse=True)
        component_names: list[str] = []
        for component in self.components:
            score_matrix = component.score_users(unique_users).astype(np.float32)
            raw = score_matrix[inverse, items]
            normed = minmax(score_matrix, axis=1)[inverse, items]
            columns.extend([raw, normed])
            component_names.extend([f"{component.name}_raw", f"{component.name}_norm"])
        if self.include_popularity:
            columns.append(self.popularity_[items])
            component_names.append("train_popularity")
        metadata = self._catalog_features(users, items)
        columns.extend(metadata.T)
        self._last_component_feature_names_ = component_names
        return np.vstack(columns).T.astype(np.float32)
    
    # tạo feature cho từng cặp (user, item) dựa trên metadata của item từ catalog,
    def _catalog_features(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        catalog = self.dataset_.catalog
        values = np.zeros((len(items), 7), dtype=np.float32)
        for row_idx, (user, item) in enumerate(zip(users, items, strict=False)):
            row = catalog.iloc[int(item)]
            values[row_idx, 0] = _float(row.get("vote_average", 0.0)) / 10.0
            values[row_idx, 1] = np.log1p(_float(row.get("vote_count", 0.0)))
            values[row_idx, 2] = np.log1p(_float(row.get("popularity", 0.0)))
            year = _float(row.get("release_year", 0.0))
            values[row_idx, 3] = (year - 1900.0) / 150.0 if year > 0 else 0.0
            values[row_idx, 4] = _float(row.get("runtime_minutes", 0.0)) / 240.0
            values[row_idx, 5] = self._genre_overlap(int(user), int(item))
            values[row_idx, 6] = np.log1p(len(self.dataset_.train_user_items.get(int(user), set())))
        return values

    def _feature_names(self) -> list[str]:
        names: list[str] = []
        for component in self.components:
            names.extend([f"{component.name}_raw", f"{component.name}_norm"])
        if self.include_popularity:
            names.append("train_popularity")
        names.extend(
            [
                "vote_average",
                "log_vote_count",
                "log_tmdb_popularity",
                "release_year_scaled",
                "runtime_scaled",
                "genre_overlap",
                "log_user_history_length",
            ]
        )
        return names

    def _build_user_genres(self, dataset) -> dict[int, set[str]]:
        result: dict[int, set[str]] = {}
        for user, items in dataset.train_user_items.items():
            genres: set[str] = set()
            for item in items:
                genres.update(self.item_genres_[int(item)])
            result[int(user)] = genres
        return result
    
    # đo phim candidate có overlap thể loại với phim mà user đã xem hay không,
    def _genre_overlap(self, user: int, item: int) -> float:
        user_genres = self.user_genres_.get(int(user), set())
        item_genres = self.item_genres_[int(item)]
        if not user_genres or not item_genres:
            return 0.0
        return float(len(user_genres & item_genres) / max(1, len(item_genres)))

# tạo các bộ trọng số có tổng = 1 -> dùng trong grid search
def _weight_grid(component_count: int, step: float = 0.25) -> list[np.ndarray]:
    if component_count <= 1:
        return [np.ones(1, dtype=np.float32)]
    if component_count > 4:
        return [np.ones(component_count, dtype=np.float32) / component_count]
    steps = max(1, int(round(1.0 / step)))
    candidates: list[np.ndarray] = []

    def build(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            candidates.append(np.asarray([*prefix, remaining], dtype=np.float32) / steps)
            return
        for value in range(remaining + 1):
            build([*prefix, value], remaining - value, slots - 1)

    build([], steps, component_count)
    return candidates


def _group_sizes(users: np.ndarray) -> list[int]:
    _, counts = np.unique(users, return_counts=True)
    return [int(count) for count in counts]

# chọn 1 item negative sample cho mỗi user, 
# đảm bảo rằng item đó chưa từng được user tương tác,
#  nếu không thể tìm được item nào thì trả về None
def _sample_negative_item(rng: np.random.Generator, num_items: int, seen: set[int]) -> int | None:
    if len(seen) >= num_items:
        return None
    for _ in range(100):
        item = int(rng.integers(0, num_items))
        if item not in seen:
            return item
    candidates = np.setdiff1d(np.arange(num_items, dtype=np.int64), np.fromiter(seen, dtype=np.int64), assume_unique=False)
    if candidates.size == 0:
        return None
    return int(rng.choice(candidates))

# lấy thể loại của item từ catalog, 
# trả về một set các thể loại đã được chuẩn hóa (lowercase, strip, 
# loại bỏ rỗng và "(no genres listed)")
def _genre_set(row) -> set[str]:
    values = f"{row.get('genres', '')}|{row.get('tmdb_genres', '')}".replace(",", "|").split("|")
    return {value.strip().lower() for value in values if value.strip() and value.strip() != "(no genres listed)"}


def _float(value: Any) -> float:
    try:
        if value is None or np.isnan(value):
            return 0.0
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
