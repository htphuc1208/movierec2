"""MovieLens loading, encoding and splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import sparse


MOVIELENS_URLS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}


@dataclass(frozen=True)
class MovieLensData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    links: pd.DataFrame


@dataclass(frozen=True)
class PreparedInteractions:
    interactions: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    # Mappings from original IDs to encoded indices
    user_mapping: Dict[int, int] 
    item_mapping: Dict[int, int]

    @property
    def num_users(self) -> int:
        return len(self.user_mapping)

    @property
    def num_items(self) -> int:
        return len(self.item_mapping)


def read_movielens(raw_dir: str | Path) -> MovieLensData:
    """Read MovieLens CSV or DAT files from a dataset directory."""
    raw_dir = Path(raw_dir)
    ratings_csv = raw_dir / "ratings.csv"
    movies_csv = raw_dir / "movies.csv"
    links_csv = raw_dir / "links.csv"

    if ratings_csv.exists() and movies_csv.exists():
        ratings = pd.read_csv(ratings_csv)
        movies = pd.read_csv(movies_csv)
        links = pd.read_csv(links_csv) if links_csv.exists() else pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    elif (raw_dir / "ratings.dat").exists() and (raw_dir / "movies.dat").exists():
        ratings = pd.read_csv(
            raw_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            raw_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        links = pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
    else:
        raise FileNotFoundError(f"MovieLens files were not found in {raw_dir}")
    
    # kiem tra du cot khong va ep kieu du lieu cho ratings va movies,
    #  ep kieu cho links neu co, tra ve MovieLensData
    required_rating_cols = {"userId", "movieId", "rating", "timestamp"}
    required_movie_cols = {"movieId", "title", "genres"}
    if not required_rating_cols.issubset(ratings.columns):
        raise ValueError(f"ratings file must include {sorted(required_rating_cols)}")
    if not required_movie_cols.issubset(movies.columns):
        raise ValueError(f"movies file must include {sorted(required_movie_cols)}")

    ratings = ratings[list(required_rating_cols)].copy()
    movies = movies[["movieId", "title", "genres"]].copy()
    links = links.copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)

    if "tmdbId" in links.columns:
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    if "imdbId" in links.columns:
        links["imdbId"] = pd.to_numeric(links["imdbId"], errors="coerce").astype("Int64")
    if "movieId" in links.columns:
        links["movieId"] = links["movieId"].astype(int)

    return MovieLensData(ratings=ratings, movies=movies, links=links)


def prepare_interactions(
    ratings: pd.DataFrame,
    min_rating: float = 4.0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> PreparedInteractions:
    """Filter positive feedback, encode IDs and split each user by timestamp."""
    # Lọc các tương tác có rating >= min_rating, nếu không còn tương tác nào thì raise lỗi
    interactions = ratings.loc[ratings["rating"] >= min_rating].copy()
    if interactions.empty:
        raise ValueError("No positive interactions remain after applying min_rating")

    # Encode userId và movieId thành user_idx và item_idx, tạo mapping từ ID gốc sang index, sắp xếp theo user_idx, timestamp, item_idx
    users = sorted(interactions["userId"].unique().tolist())
    items = sorted(interactions["movieId"].unique().tolist())
    user_mapping = {int(user_id): idx for idx, user_id in enumerate(users)}
    item_mapping = {int(movie_id): idx for idx, movie_id in enumerate(items)}
    # Thêm cột user_idx và item_idx vào interactions bằng cách map userId và movieId qua user_mapping và item_mapping, ép kiểu int, sắp xếp theo user_idx, timestamp, item_idx, reset index
    interactions["user_idx"] = interactions["userId"].map(user_mapping).astype(int)
    interactions["item_idx"] = interactions["movieId"].map(item_mapping).astype(int)
    interactions = interactions.sort_values(["user_idx", "timestamp", "item_idx"]).reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # Với mỗi user_idx, lấy các tương tác của user đó, sắp xếp theo timestamp và item_idx, chia thành train/val/test theo test_ratio và val_ratio, đảm bảo mỗi phần có ít nhất 1 tương tác và train có ít nhất 3 tương tác nếu có thể
    for _, group in interactions.groupby("user_idx", sort=False):
        group = group.sort_values(["timestamp", "item_idx"])
        n = len(group)
        # Nếu user có ít hơn 3 tương tác, bỏ qua việc chia và đưa tất cả vào train
        if n < 3:
            train_parts.append(group)
            continue
        # Tính số lượng tương tác cho test và val, đảm bảo mỗi phần có ít nhất 1 tương tác và train có ít nhất 3 tương tác nếu có thể
        test_count = max(1, int(round(n * test_ratio)))
        val_count = max(1, int(round(n * val_ratio))) if n - test_count >= 3 else 0
        train_end = max(1, n - val_count - test_count)
        val_end = n - test_count

        train_parts.append(group.iloc[:train_end])
        if val_count:
            val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0].copy()
    val = pd.concat(val_parts, ignore_index=True) if val_parts else interactions.iloc[0:0].copy()
    test = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0].copy()

    return PreparedInteractions(
        interactions=interactions,
        train=train,
        val=val,
        test=test,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
    )


def build_user_item_sets(interactions: pd.DataFrame) -> dict[int, set[int]]:
    """Build user -> interacted item set from encoded interactions."""
    result: dict[int, set[int]] = {}
    for row in interactions[["user_idx", "item_idx"]].itertuples(index=False):
        result.setdefault(int(row.user_idx), set()).add(int(row.item_idx))
    return result


def build_sparse_interaction_matrix(
    interactions: pd.DataFrame,
    num_users: int,
    num_items: int,
    value_col: str | None = None,
) -> sparse.csr_matrix:
    """Build a user-item CSR matrix from encoded interactions."""
    values = (
        interactions[value_col].to_numpy(dtype=np.float32)
        if value_col and value_col in interactions.columns
        else np.ones(len(interactions), dtype=np.float32)
    )
    return sparse.csr_matrix(
        (
            values,
            (
                interactions["user_idx"].to_numpy(dtype=np.int64),
                interactions["item_idx"].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(num_users, num_items),
    )


def filter_catalog_to_items(movies: pd.DataFrame, item_mapping: dict[int, int]) -> pd.DataFrame:
    """Return a catalog ordered by encoded item index."""
    ordered = pd.DataFrame({"movieId": list(item_mapping.keys()), "item_idx": list(item_mapping.values())})
    catalog = ordered.merge(movies, on="movieId", how="left").sort_values("item_idx").reset_index(drop=True)
    catalog["title"] = catalog["title"].fillna(catalog["movieId"].map(lambda mid: f"Movie {mid}"))
    catalog["genres"] = catalog["genres"].fillna("")
    return catalog


def external_ids_from_context(values: Iterable[str | int]) -> list[int]:
    """Parse MovieLens IDs from API session context values."""
    parsed: list[int] = []
    for value in values:
        text = str(value)
        if text.startswith("ml_"):
            text = text[3:]
        if text.isdigit():
            parsed.append(int(text))
    return parsed
