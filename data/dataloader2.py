from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# =====================================================================
# PHẦN 1: BỘ GOM VÀ XỬ LÝ DỮ LIỆU GỐC (STANDALONE)
# =====================================================================

@dataclass(frozen=True)
class MovieDataBundle:
    movies: pd.DataFrame
    ratings: pd.DataFrame
    tags: pd.DataFrame
    links: pd.DataFrame


class MovieLensDataLoader:
    """Tải các tệp CSV kiểu MovieLens và siêu dữ liệu mở rộng enriched_movies."""

    def __init__(self, data_dir: str | Path = "data/sample") -> None:
        # data_dir sẽ được truyền từ file chạy bên ngoài vào
        self.data_dir = Path(data_dir)

    def load(self) -> MovieDataBundle:
        return MovieDataBundle(
            movies=self.load_movies(),
            ratings=self.load_ratings(),
            tags=self.load_tags(),
            links=self.load_links(),
        )

    def load_movies(self) -> pd.DataFrame:
        movies = self._read_required_csv("movies.csv")
        movies["movieId"] = movies["movieId"].astype(int)
        movies["title"] = movies["title"].astype(str)
        movies["genres"] = movies["genres"].fillna("").astype(str)

        if "year" not in movies.columns:
            movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").fillna("")

        links_path = self.data_dir / "links.csv"
        if links_path.exists():
            links = pd.read_csv(links_path)
            links["movieId"] = links["movieId"].astype(int)
            movies = movies.merge(links, on="movieId", how="left")

        enriched_path = self.data_dir / "enriched_movies.csv"
        if enriched_path.exists():
            enriched = pd.read_csv(enriched_path)
            enriched["movieId"] = enriched["movieId"].astype(int)
            movies = movies.merge(enriched, on="movieId", how="left", suffixes=("", "_enriched"))
            for column in ["overview", "tagline", "director", "cast", "poster_url", "budget", "revenue"]:
                enriched_column = f"{column}_enriched"
                if enriched_column in movies.columns:
                    movies[column] = movies[column].combine_first(movies[enriched_column])
                    movies = movies.drop(columns=[enriched_column])

        for column in ["overview", "tagline", "director", "cast", "poster_url"]:
            if column not in movies.columns:
                movies[column] = ""
            movies[column] = movies[column].fillna("").astype(str)

        for column in ["tmdbId", "imdbId"]:
            if column not in movies.columns:
                movies[column] = ""
            movies[column] = movies[column].fillna("")

        return movies

    def load_ratings(self) -> pd.DataFrame:
        ratings = self._read_required_csv("ratings.csv")
        ratings["userId"] = ratings["userId"].astype(int)
        ratings["movieId"] = ratings["movieId"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)
        if "timestamp" not in ratings.columns:
            ratings["timestamp"] = range(len(ratings))
        return ratings

    def load_tags(self) -> pd.DataFrame:
        path = self.data_dir / "tags.csv"
        if not path.exists():
            return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        tags = pd.read_csv(path)
        if tags.empty:
            return pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        tags["movieId"] = tags["movieId"].astype(int)
        tags["tag"] = tags["tag"].fillna("").astype(str)
        return tags

    def load_links(self) -> pd.DataFrame:
        path = self.data_dir / "links.csv"
        if not path.exists():
            return pd.DataFrame(columns=["movieId", "imdbId", "tmdbId"])
        links = pd.read_csv(path)
        links["movieId"] = links["movieId"].astype(int)
        return links

    def train_val_test_split(
        self,
        ratings: pd.DataFrame,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Chia nhỏ các tương tác của từng user theo timestamp nhằm chống rò rỉ tương lai."""

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

        return (
            self._concat_or_empty(train_parts, ratings.columns),
            self._concat_or_empty(val_parts, ratings.columns),
            self._concat_or_empty(test_parts, ratings.columns),
        )

    @staticmethod
    def build_implicit_interactions(
        ratings: pd.DataFrame,
        positive_threshold: float = 4.0,
    ) -> pd.DataFrame:
        interactions = ratings.loc[ratings["rating"] >= positive_threshold, ["userId", "movieId", "rating"]]
        return interactions.drop_duplicates(["userId", "movieId"]).reset_index(drop=True)

    @staticmethod
    def require_columns(frame: pd.DataFrame, required: Iterable[str], source: str) -> None:
        missing = [column for column in required if column not in frame.columns]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"{source} is missing required columns: {joined}")

    def _read_required_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _concat_or_empty(parts: list[pd.DataFrame], columns: Iterable[str]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(parts, ignore_index=True)


# =====================================================================
# PHẦN 2: BỘ ĐÓNG GÓI PYTORCH DATASET CHO KIẾN TRÚC THÁP ĐÔI
# Chức năng: Đọc ma trận nhúng SBERT và tự động sinh Cặp Âm - Dương (Triplet).
# =====================================================================

class TwoTowerDataset(Dataset):
    def __init__(self, ratings_df: pd.DataFrame, embeddings_path: str, mapping_path: str):
        super().__init__()
        print(f"[Dataset] Đang nạp ma trận nhúng SBERT từ: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        self.movie_map = np.load(mapping_path, allow_pickle=True).item()
        
        self.good_ratings = ratings_df[ratings_df["rating"] >= 4.0]
        self.user_history = self.good_ratings.groupby("userId")["movieId"].apply(set).to_dict()
        self.all_movie_ids = list(self.movie_map.keys())
        
        print("[Dataset] Đang pre-compute User Profile (Tính trung bình vector đặc trưng)...")
        self.user_profiles = {}
        for user_id, movie_set in self.user_history.items():
            valid_indices = [self.movie_map[m] for m in movie_set if m in self.movie_map]
            if valid_indices:
                self.user_profiles[user_id] = np.mean(self.embeddings[valid_indices], axis=0)
            else:
                self.user_profiles[user_id] = np.zeros(self.embeddings.shape[1])

        valid_interactions = self.good_ratings[self.good_ratings['movieId'].isin(self.all_movie_ids)]
        self.interactions = list(zip(valid_interactions['userId'], valid_interactions['movieId']))
        print(f"[Dataset] Đã tạo xong {len(self.interactions)} cặp tương tác tích cực (Mẫu Dương).")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        user_id, pos_movie_id = self.interactions[idx]
        user_vec = self.user_profiles[user_id]
        
        pos_idx = self.movie_map[pos_movie_id]
        pos_vec = self.embeddings[pos_idx]
        
        neg_movie_id = random.choice(self.all_movie_ids)
        while neg_movie_id in self.user_history.get(user_id, set()):
            neg_movie_id = random.choice(self.all_movie_ids)
            
        neg_idx = self.movie_map[neg_movie_id]
        neg_vec = self.embeddings[neg_idx]
        
        return (
            torch.tensor(user_vec, dtype=torch.float32),
            torch.tensor(pos_vec, dtype=torch.float32),
            torch.tensor(neg_vec, dtype=torch.float32)
        )


# =====================================================================
# PHẦN 3: HÀM SẢN XUẤT DATALOADER PHỤC VỤ KỊCH BẢN HUẤN LUYỆN
# =====================================================================
def get_twotower_dataloaders(data_dir='../data/sample', artifact_dir='../artifacts', batch_size=32):
    """
    Hàm gọi trích xuất, tự động khớp đường dẫn dựa trên vị trí file chạy Notebook.
    Mặc định: khi chạy từ models/train_content.ipynb, CWD là models/ nên đường dẫn
    vẫn cần lùi 1 cấp để ra ngoài tìm data và artifacts.
    """
    loader = MovieLensDataLoader(data_dir=data_dir)
    bundle = loader.load()
    
    print("[DataLoader2] Tiến hành kích hoạt bộ phân rã chia tách Train/Val/Test theo dòng thời gian...")
    train_df, val_df, test_df = loader.train_val_test_split(bundle.ratings)
    
    emb_path = f"{artifact_dir}/movie_embeddings.npy"
    map_path = f"{artifact_dir}/movie_id_map.npy"
    
    train_dataset = TwoTowerDataset(train_df, emb_path, map_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_df, test_df