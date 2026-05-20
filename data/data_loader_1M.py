from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import pandas as pd


@dataclass(frozen=True)
class MovieDataBundle:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


class MovieLensDataLoader:
    """Load, process, map continuous IDs, and save MovieLens 1M dataset."""

    def __init__(self, data_dir: str | Path = "data/raw", processed_dir: str | Path = "data/processed") -> None:
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)

    def load_and_save(self) -> MovieDataBundle:
        """
        Khép kín quy trình: Nạp Raw -> Tạo Mapping Index đồng bộ -> 
        Áp dụng cho 3 bảng -> Lưu vật lý ra thư mục Processed -> Đóng gói.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Nạp dữ liệu thô (đã xử lý kiểu dữ liệu, làm sạch text)
        movies = self.load_movies()
        ratings = self.load_ratings()
        users = self.load_users()
        tags = self.load_tags()
        links = self.load_links()

        # 2. Xây dựng bộ Mapping ID liên tục (0 -> N-1) từ bảng thực thể gốc
        # Sắp xếp theo ID gốc để index sinh ra luôn đồng nhất mỗi lần chạy
        movies = movies.sort_values("movieId").reset_index(drop=True)
        users = users.sort_values("userId").reset_index(drop=True)

        movie_to_idx = {mid: idx for idx, mid in enumerate(movies["movieId"])}
        user_to_idx = {uid: idx for idx, uid in enumerate(users["userId"])}

        # 3. Chèn Index vào bảng Movies và Users (Chèn lên đầu cho dễ nhìn)
        movies.insert(0, "movie_idx", movies["movieId"].map(movie_to_idx))
        users.insert(0, "user_idx", users["userId"].map(user_to_idx))

        # 4. Áp dụng Mapping đó lên bảng Ratings
        ratings.insert(0, "movie_idx", ratings["movieId"].map(movie_to_idx))
        ratings.insert(0, "user_idx", ratings["userId"].map(user_to_idx))

        # Dọn dẹp phòng hờ dữ liệu lỗi (những ID không tồn tại trong bảng gốc)
        ratings = ratings.dropna(subset=["user_idx", "movie_idx"])
        ratings["user_idx"] = ratings["user_idx"].astype(int)
        ratings["movie_idx"] = ratings["movie_idx"].astype(int)

        # 5. Đóng gói vào Bundle
        bundle = MovieDataBundle(
            movies=movies,
            ratings=ratings,
            users=users,
            tags=tags,
            links=links,
        )
        
        self._save_bundle(bundle)
        return bundle

    def load_movies(self) -> pd.DataFrame:
        dat_path = self.data_dir / "movies.dat"
        if not dat_path.exists():
            raise FileNotFoundError(f"Missing required 1M data file: {dat_path}")

        movies = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="ISO-8859-1",
        )

        movies["movieId"] = movies["movieId"].astype(int)
        movies["title"] = movies["title"].astype(str)
        movies["genres"] = movies["genres"].fillna("").astype(str)
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").fillna("0").astype(int)
        movies["title"] = movies["title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)\s*$", "", x))

        for column in ["overview", "tagline", "director", "cast", "poster_url", "tmdbId", "imdbId"]:
            movies[column] = ""

        return movies

    def load_ratings(self) -> pd.DataFrame:
        dat_path = self.data_dir / "ratings.dat"
        if not dat_path.exists():
            raise FileNotFoundError(f"Missing required 1M data file: {dat_path}")

        ratings = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="ISO-8859-1",
        )

        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        ratings["timestamp"] = ratings["timestamp"].astype(int)
        
        return ratings

    def load_users(self) -> pd.DataFrame:
        dat_path = self.data_dir / "users.dat"
        if not dat_path.exists():
            raise FileNotFoundError(f"Missing required 1M data file: {dat_path}")

        users = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zipCode"],
            encoding="ISO-8859-1",
        )

        users["userId"] = users["userId"].astype(int)
        users["gender"] = users["gender"].astype(str)
        users["age"] = users["age"].astype(int)
        users["occupation"] = users["occupation"].astype(int)
        users["zipCode"] = users["zipCode"].astype(str)

        return users

    def load_tags(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    def load_links(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])

    def _save_bundle(self, bundle: MovieDataBundle) -> None:
        print(f"Đang tự động lưu dữ liệu sạch vào cấu trúc thư mục: {self.processed_dir}...")
        
        bundle.movies.to_csv(self.processed_dir / "movies_clean.csv", index=False, encoding="utf-8")
        bundle.ratings.to_csv(self.processed_dir / "ratings_clean.csv", index=False, encoding="utf-8")
        bundle.users.to_csv(self.processed_dir / "users_clean.csv", index=False, encoding="utf-8")
        bundle.tags.to_csv(self.processed_dir / "tags_clean.csv", index=False, encoding="utf-8")
        
        movie_metadata = bundle.movies[["movie_idx", "movieId", "title", "overview", "poster_url"]]
        movie_metadata.to_csv(self.processed_dir / "movie_metadata.csv", index=False, encoding="utf-8")
        
        print("Lưu tệp thành công! Dữ liệu đã sẵn sàng cho Model học sâu (Deep Learning).")

    def train_val_test_split(
        self,
        ratings: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        save_splits: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Đang phân tách tập Train/Val/Test theo trục thời gian...")
        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for _, user_rows in ratings.sort_values("timestamp").groupby("userId"):
            count = len(user_rows)
            if count < 3:
                train_parts.append(user_rows)
                continue
            test_size = max(1, int(round(count * test_ratio)))
            val_size = max(1, int(round(count * val_ratio)))
            train_end = max(1, count - val_size - test_size)
            val_end = count - test_size
            
            train_parts.append(user_rows.iloc[:train_end])
            val_parts.append(user_rows.iloc[train_end:val_end])
            test_parts.append(user_rows.iloc[val_end:])

        train_df = self._concat_or_empty(train_parts, ratings.columns)
        val_df = self._concat_or_empty(val_parts, ratings.columns)
        test_df = self._concat_or_empty(test_parts, ratings.columns)

        if save_splits:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(self.processed_dir / "train_ratings.csv", index=False, encoding="utf-8")
            val_df.to_csv(self.processed_dir / "val_ratings.csv", index=False, encoding="utf-8")
            test_df.to_csv(self.processed_dir / "test_ratings.csv", index=False, encoding="utf-8")
            print("Đã lưu các tập phân rã: train_ratings.csv, val_ratings.csv, test_ratings.csv")

        return train_df, val_df, test_df

    @staticmethod
    def build_implicit_interactions(
        ratings: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> pd.DataFrame:
        interactions = ratings.loc[ratings["rating"] >= positive_threshold, ["user_idx", "movie_idx", "rating"]]
        return interactions.drop_duplicates(["user_idx", "movie_idx"]).reset_index(drop=True)

    @staticmethod
    def _concat_or_empty(parts: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(parts, ignore_index=True)