from __future__ import annotations

import os
import pandas as pd
from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import MovieLensDataLoader

from datetime import datetime

# Nhập các class Model từ thư mục src/models của dự án
from models import HybridMovieRecommender

# ==========================================
# 1. Định nghĩa Data Models (Pydantic)
# ==========================================
class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=10, ge=1, le=50)
    model_type: str = Field(default="hybrid", description="svd | lightgcn | twotower | hybrid")
    w_cf: float = Field(default=0.5, description="Trọng số w1 cho LightGCN (Collaborative)")
    w_cb: float = Field(default=0.5, description="Trọng số w2 cho Two-Tower (Content-based)")
    exclude_seen: bool = True

class RecommendResponse(BaseModel):
    model_used: str
    recommendations: list[dict[str, Any]]

# ==========================================
# 2. Khởi tạo FastAPI
# ==========================================
app = FastAPI(
    title="Hybrid Recommender API",
    version="1.0.0",
    description="Hệ thống Gợi ý sử dụng SVD, LightGCN và Two-Tower Model.",
)

# 1. Hàm load toàn bộ Data (Có cache để không đọc lại file)
@lru_cache(maxsize=1)
def get_data_bundle():
    data_dir = os.getenv("MOVIEREC_DATA_DIR", "data/sample")
    return MovieLensDataLoader(data_dir).load()

# 2. Cập nhật hàm khởi tạo model (Lấy data từ bundle ở trên)
@lru_cache(maxsize=1)
def get_recommender() -> HybridMovieRecommender:
    bundle = get_data_bundle()
    model = HybridMovieRecommender()
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    return model

# 3. Cập nhật lại hàm lấy Full Data phim
@lru_cache(maxsize=1)
def get_full_movies_data() -> list[dict[str, Any]]:
    return get_data_bundle().movies.to_dict(orient="records")

# ==========================================
# 3. Các API Routes
# ==========================================
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/users")
def users() -> dict[str, list[int]]:
    # Lấy danh sách user thẳng từ bảng ratings của DataLoader
    bundle = get_data_bundle()
    unique_users = sorted(bundle.ratings["userId"].unique().tolist())
    return {"users": unique_users}

@app.get("/users/{user_id}/history")
def user_history(user_id: int) -> dict[str, list[dict[str, Any]]]:
    bundle = get_data_bundle()
    ratings_df = bundle.ratings
    movies_df = bundle.movies

    # 1. Lọc rating của đúng user_id này
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        return {"history": []}

    # 2. Ghép với bảng movies để lấy tên phim và thể loại
    history_df = user_ratings.merge(movies_df, on='movieId', how='left')
    
    # 3. Sắp xếp mới nhất lên đầu
    history_df = history_df.sort_values(by='timestamp', ascending=False)

    # 4. Format dữ liệu trả về cho Streamlit (Khớp với format của thẻ phim)
    history_list = []
    for _, row in history_df.iterrows():
        history_list.append({
            "movieId": row['movieId'],
            "title": row.get("title", f"Phim ID {row['movieId']}"),
            "score": row['rating'], # Tái sử dụng trường score để hiển thị điểm đánh giá
            "genres": str(row.get("genres", "")).replace("|", ", "),
            "poster_url": row.get("poster_url", "")
        })

    return {"history": history_list}
    
@app.get("/filters")
def get_filters() -> dict[str, list[Any]]:
    try:
        genres, years = get_recommender().get_available_filters()
        return {"genres": genres, "years": years}
    except AttributeError:
        # ============================================
        # Dữ liệu giả lập (mock) nếu chưa code hàm lọc
        # ============================================
        return {
            "genres": ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Thriller"],
            "years": [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]
        }

@app.get("/movies")
def movies(
    genre: Optional[str] = None,
    year: Optional[int] = None,
    search: Optional[str] = None
) -> dict[str, list[dict[str, Any]]]:
    
    all_movies = get_recommender().movies_for_picker()
    
    if genre:
        all_movies = [m for m in all_movies if genre.lower() in m.get("genres", "").lower()]
    if year:
        all_movies = [m for m in all_movies if str(year) == str(m.get("year", ""))]
    if search:
        all_movies = [m for m in all_movies if search.lower() in m.get("title", "").lower()]
        
    # Mặc định trả về phim trending hoặc nguyên bản nếu không lọc
    if not genre and not year and not search:
        all_movies = sorted(all_movies, key=lambda x: x.get("rating_count", 0), reverse=True)
        
    return {"movies": all_movies}

@app.get("/movies/{movie_id}")
def movie_details(movie_id: str) -> dict[str, Any]:
    # Gọi hàm mới để lấy FULL dữ liệu thay vì dùng get_recommender().movies_for_picker()
    all_movies = get_full_movies_data()
    
    for movie in all_movies:
        if str(movie.get("movieId")) == str(movie_id):
            cleaned_movie = {}
            for k, v in movie.items():
                val_str = str(v).strip().lower()
                if pd.isna(v) or val_str == "nan" or val_str == "":
                    cleaned_movie[k] = "Chưa có thông tin tóm tắt." if k == "overview" else "Đang cập nhật"
                else:
                    cleaned_movie[k] = v
                    
            return cleaned_movie
            
    raise HTTPException(status_code=404, detail="Không tìm thấy phim")

@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    try:
        similar = get_recommender().get_similar_items_two_tower(movie_id, top_k)
        return {"similar": similar}
    except AttributeError:
        # Bắt lỗi khi chưa code hàm get_similar_items_two_tower
        return {"similar": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    recommender = get_recommender()
    recs = []
    
    try:
        if request.model_type == "svd":
            recs = recommender.predict_svd(request.user_id, request.top_k, request.exclude_seen)
        elif request.model_type == "lightgcn":
            recs = recommender.predict_lightgcn(request.user_id, request.top_k, request.exclude_seen)
        elif request.model_type == "twotower":
            recs = recommender.predict_two_tower(request.user_id, request.top_k, request.exclude_seen)
        elif request.model_type == "hybrid":
            recs = recommender.predict_hybrid(
                user_id=request.user_id, 
                top_k=request.top_k, 
                w_cf=request.w_cf, 
                w_cb=request.w_cb,
                exclude_seen=request.exclude_seen
            )
        else:
            raise ValueError("model_type không hợp lệ.")
            
    except AttributeError:
        # Tạm thời trả về mảng rỗng khi model chưa hoàn thiện
        return RecommendResponse(model_used=request.model_type, recommendations=[])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
        
    return RecommendResponse(model_used=request.model_type, recommendations=recs)