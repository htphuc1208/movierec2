from __future__ import annotations

import os
import re
import pandas as pd
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data import MovieLensDataLoader
from models import HybridMovieRecommender
from models import SBERTRecommender

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"AI đang chạy trên thiết bị: {device}")

from models.SVD import SVDModel

# ---------------------------------------------------------
# KHỞI TẠO DEEP LEARNING (SVD MODEL)
# ---------------------------------------------------------
svd_model = None
try:
    if os.path.exists("artifacts/svd_baseline.pt"):
        # Đọc gói Checkpoint (tắt weights_only)
        checkpoint = torch.load("artifacts/svd_baseline.pt", map_location='cpu', weights_only=False)
        
        # Lấy model_state_dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        num_users = state_dict['user_embedding.weight'].shape[0]
        num_items = state_dict['item_embedding.weight'].shape[0]
        emb_dim = state_dict['user_embedding.weight'].shape[1]
        
        global_mean = state_dict.get('global_mean', torch.tensor(0.0)).item()
        
        svd_model = SVDModel(num_users=num_users, num_items=num_items, embedding_dim=emb_dim, global_mean=global_mean)
        svd_model.load_state_dict(state_dict)
        svd_model.eval()
        print(f"Đã nạp thành công SVD Model! (Sẵn sàng phục vụ {num_users} Users)")
except Exception as e:
    print(f"Không thể load SVD Model: {e}")


from models.LightGCN import LightGCNModel

# ---------------------------------------------------------
# KHỞI TẠO LIGHTGCN (GRAPH NEURAL NETWORK)
# ---------------------------------------------------------
lightgcn_model = None
try:
    if os.path.exists("artifacts/lightgcn_baseline.pt"):
        # Đọc file weights
        checkpoint = torch.load("artifacts/lightgcn_baseline.pt", map_location='cpu', weights_only=False)
        gcn_state = checkpoint.get('model_state_dict', checkpoint)
        
        # Trích xuất kích thước mảng
        num_users_gcn = gcn_state['user_embedding.weight'].shape[0]
        num_items_gcn = gcn_state['item_embedding.weight'].shape[0]
        emb_dim_gcn = gcn_state['user_embedding.weight'].shape[1]
        
        # Khởi tạo Model
        lightgcn_model = LightGCNModel(num_users_gcn, num_items_gcn, embedding_dim=emb_dim_gcn)
        lightgcn_model.load_state_dict(gcn_state)
        lightgcn_model.eval()
        
        print(f"Đã nạp thành công LightGCN Model!")
except Exception as e:
    print(f"Không thể load LightGCN Model: {e}")

# ---------------------------------------------------------
# KHỞI TẠO TWO-TOWER (SBERT)
# ---------------------------------------------------------
try:
    movies_df = pd.read_csv("data/ml-latest-small/movies.csv") 
    tags_df = pd.read_csv("data/ml-latest-small/tags.csv") 
    
    content_recommender = SBERTRecommender()
    print("Đang tải mạng Nơ-ron ngôn ngữ SBERT...")
    content_recommender.fit(movies_df, tags=tags_df)
    print("SBERT Build xong! Đã load dữ liệu ngôn ngữ kho phim.")
except Exception as e:
    print(f"LỖI KHỞI TẠO SBERT: {e}")
    content_recommender = None


class RecommendRequest(BaseModel):
    user_id: int | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    session_context: list[str] = Field(default_factory=list)
    exclude_seen: bool = True
    model_name: str = "SVD"


class RecommendResponse(BaseModel):
    recommendations: list[dict[str, Any]]


class ModelInfoResponse(BaseModel):
    model_info: dict[str, Any]


app = FastAPI(
    title="Hybrid Movie Recommendation API",
    version="0.1.0",
    description="MovieLens-style hybrid recommender with collaborative and metadata signals.",
)


@lru_cache(maxsize=1)
def get_recommender() -> HybridMovieRecommender:
    data_dir = os.getenv("MOVIEREC_DATA_DIR", "data/ml-latest-small")
    bundle = MovieLensDataLoader(data_dir).load()
    model = HybridMovieRecommender()
    artifact_dir = os.getenv("MOVIEREC_ARTIFACT_DIR", "").strip()
    if artifact_dir:
        try:
            model.load_artifact(artifact_dir, bundle.movies, bundle.ratings, bundle.tags)
            if not model.dataset_name:
                model.dataset_name = data_dir
            return model
        except Exception as exc:
            model.artifact_manifest = {"artifact_load_error": str(exc), "artifact_path": artifact_dir}
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    model.dataset_name = data_dir
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users")
def users() -> dict[str, list[int]]:
    return {"users": get_recommender().users()}

@app.get("/users/{user_id}/history")
def get_user_history(user_id: int, top_k: int = 15):
    try:
        # Lấy toàn bộ dữ liệu phim hiện có
        all_movies = get_movies_with_scores()
        movie_dict = {int(float(m.get("movie_id", m.get("movieId", m.get("id"))))): m for m in all_movies if m.get("movie_id", m.get("movieId", m.get("id")))}
        
        # Lấy dữ liệu rating
        recommender = get_recommender()
        if not hasattr(recommender, 'ratings') or recommender.ratings is None:
            return {"movies": []}
            
        # Lọc các phim user đã rate
        user_history = recommender.ratings[recommender.ratings['userId'] == user_id]
        
        # Sắp xếp theo điểm rating giảm dần
        user_history = user_history.sort_values(by='rating', ascending=False).head(top_k)
        
        # Map ID phim với thông tin chi tiết
        history_movies = []
        for _, row in user_history.iterrows():
            m_id = int(row['movieId'])
            if m_id in movie_dict:
                m_copy = movie_dict[m_id].copy()
                # Gắn thêm điểm user đã đánh giá
                m_copy['user_rating'] = float(row['rating'])
                history_movies.append(m_copy)
                
        return {"movies": history_movies}
        
    except Exception as e:
        print(f"Lỗi khi lấy lịch sử user: {e}")
        return {"movies": []}


# ---------------------------------------------------------
# ĐỌC RATINGS TRỰC TIẾP TỪ CSV
# ---------------------------------------------------------
def get_movies_with_scores() -> list[dict[str, Any]]:
    model = get_recommender()
    all_movies = model.movies_for_picker()
    
    score_map, count_map = {}, {}
    # Đọc trực tiếp từ file csv để tránh lỗi vote = 0
    try:
        ratings_path = "data/ml-latest-small/ratings.csv"
        if os.path.exists(ratings_path):
            ratings_df = pd.read_csv(ratings_path)
            stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count'])
            score_map = stats['mean'].to_dict()
            count_map = stats['count'].to_dict()
    except Exception as e:
        print(f"Không thể tính toán điểm từ ratings.csv: {e}")

    for m in all_movies:
        raw_id = m.get("movie_id", m.get("movieId", m.get("id")))
        m_id = 0
        if raw_id is not None and str(raw_id).strip() != "":
            try: m_id = int(float(raw_id))
            except ValueError: pass
                
        m["score"] = float(score_map.get(m_id, 0.0))
        m["vote_count"] = int(count_map.get(m_id, 0))
        
        title = str(m.get("title", ""))
        year_match = re.search(r'\((\d{4})\)', title)
        m["year"] = int(year_match.group(1)) if year_match else 1900
        
    return all_movies

@app.get("/movies")
def movies(search: str | None = None) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    if search:
        search_lower = search.lower()
        filtered_movies = [m for m in all_movies if search_lower in str(m.get("title", "")).lower()]
        return {"movies": filtered_movies}
    return {"movies": all_movies}

@app.get("/movies/trending")
def trending_movies(top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    valid_movies = [m for m in all_movies if m.get("vote_count", 0) >= 15]
    sorted_movies = sorted(valid_movies, key=lambda x: x.get("vote_count", 0), reverse=True)
    return {"movies": sorted_movies[:top_k]}

@app.get("/movies/top-rated")
def top_rated_movies(top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    valid_movies = [m for m in all_movies if m.get("vote_count", 0) >= 15]
    sorted_movies = sorted(valid_movies, key=lambda x: x.get("score", 0.0), reverse=True)
    return {"movies": sorted_movies[:top_k]}

@app.get("/movies/latest")
def latest_movies(top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    sorted_movies = sorted(all_movies, key=lambda x: (x.get("year", 0), x.get("vote_count", 0)), reverse=True)
    return {"movies": sorted_movies[:top_k]}

@app.get("/movies/genre/{genre}")
def genre_movies(genre: str, top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    filtered = [m for m in all_movies if genre.lower() in str(m.get("genres", "")).lower()]
    sorted_movies = sorted(filtered, key=lambda x: x["score"], reverse=True)
    return {"movies": sorted_movies[:top_k]}

@app.get("/movies/{movie_id}")
def movie_details(movie_id: str) -> dict[str, Any]:
    movies_df = get_recommender().movies
    matched_movie = movies_df[movies_df['movieId'].astype(str) == str(movie_id)]
    
    if matched_movie.empty:
        raise HTTPException(status_code=404, detail="Không tìm thấy phim")
        
    movie_dict = matched_movie.iloc[0].to_dict()
    cleaned_movie = {}
    for k, v in movie_dict.items():
        if pd.isna(v) or str(v).strip().lower() in ["nan", ""]:
            if k == "overview":
                cleaned_movie[k] = "Chưa có thông tin tóm tắt cho bộ phim này."
            else:
                cleaned_movie[k] = "Đang cập nhật"
        else:
            cleaned_movie[k] = v
    return cleaned_movie

@app.get("/movies/{movie_id}/similar")
def similar_movies(movie_id: str, top_k: int = 15) -> dict[str, list[dict[str, Any]]]:
    all_movies = get_movies_with_scores()
    movie_dict = {}
    for m in all_movies:
        raw_id = m.get("movie_id", m.get("movieId", m.get("id")))
        if raw_id and str(raw_id).strip() != "":
            try: movie_dict[str(int(float(raw_id)))] = m
            except ValueError: pass

    target_str_id = str(int(float(movie_id))) if movie_id else ""

    try:
        raw_recs = content_recommender.recommend_similar_movies(int(movie_id), top_k=top_k)
        if raw_recs:
            final_recs = [movie_dict[str(r["movie_id"])] for r in raw_recs if str(r["movie_id"]) in movie_dict]
            return {"movies": final_recs} 
    except Exception as e:
        print(f"AI Model chưa sẵn sàng, kích hoạt Fallback: {e}")
        pass

    if target_str_id not in movie_dict:
        return {"movies": []}
        
    target_movie = movie_dict[target_str_id]
    raw_genres = str(target_movie.get("genres", "")).replace("|", ",")
    target_genres = set([g.strip() for g in raw_genres.split(",") if g.strip()])
    
    def get_genre_score(m):
        m_raw = str(m.get("genres", "")).replace("|", ",")
        m_genres = set([g.strip() for g in m_raw.split(",") if g.strip()])
        return len(target_genres.intersection(m_genres))

    fallback_list = []
    for m in all_movies:
        m_str_id = str(int(float(m.get("movie_id", m.get("movieId", m.get("id"))))))
        if m_str_id != target_str_id:
            m_copy = m.copy()
            m_copy['temp_score'] = get_genre_score(m)
            fallback_list.append(m_copy)

    sorted_fallback = sorted(
        fallback_list, 
        key=lambda x: (x['temp_score'], x.get('vote_count', 0), x.get('score', 0.0)), 
        reverse=True
    )
    return {"movies": sorted_fallback[:top_k]}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    info = get_recommender().model_info()
    if get_recommender().artifact_manifest.get("artifact_load_error"):
        info["artifact_load_error"] = get_recommender().artifact_manifest["artifact_load_error"]
    return ModelInfoResponse(model_info=info)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    try:
        user_id = request.user_id
        # Mặc định model là SVD
        model_choice = getattr(request, 'model_name', 'SVD')

        # Tránh ghi đè dữ liệu gốc
        all_movies = get_movies_with_scores()
        movie_truth = {}
        for m in all_movies:
            raw_id = m.get("movie_id", m.get("movieId", m.get("id")))
            if raw_id:
                try: movie_truth[int(float(raw_id))] = m
                except: pass

        final_recs = []
        predictions = None
        candidate_indices = []

        # CHẠY MODEL TƯƠNG ỨNG
        if user_id is not None:
            
            # CHẠY LIGHTGCN
            if model_choice == "LightGCN" and lightgcn_model is not None and user_id < lightgcn_model.num_users:
                for m_id in movie_truth.keys():
                    if m_id < lightgcn_model.num_items:
                        candidate_indices.append(m_id)
                
                if candidate_indices:
                    with torch.no_grad():
                        u_tensor = torch.tensor([user_id] * len(candidate_indices), dtype=torch.long)
                        i_tensor = torch.tensor(candidate_indices, dtype=torch.long)
                        u_emb = lightgcn_model.user_embedding(u_tensor)
                        i_emb = lightgcn_model.item_embedding(i_tensor)
                        # Tính điểm (Dot Product)
                        predictions = (u_emb * i_emb).sum(dim=1).numpy()
                        
            # CHẠY SVD
            elif model_choice == "SVD" and svd_model is not None and user_id < svd_model.num_users:
                for m_id in movie_truth.keys():
                    if m_id < svd_model.num_items:
                        candidate_indices.append(m_id)
                
                if candidate_indices:
                    with torch.no_grad():
                        u_tensor = torch.tensor([user_id] * len(candidate_indices), dtype=torch.long)
                        i_tensor = torch.tensor(candidate_indices, dtype=torch.long)
                        predictions = svd_model(u_tensor, i_tensor).numpy()

        # QUY ĐỔI ĐIỂM SANG % MATCH
        if predictions is not None and len(candidate_indices) > 0:
            # Tìm min, max để Scale điểm
            min_p, max_p = float(predictions.min()), float(predictions.max())
            
            temp_list = []
            for i, m_id in enumerate(candidate_indices):
                m_copy = movie_truth[m_id].copy() 
                pred = float(predictions[i])
                
                # Min-Max Scaling (0.6 - 0.99)
                if max_p > min_p:
                    pct = ((pred - min_p) / (max_p - min_p)) * 0.39 + 0.60
                else:
                    pct = 0.85
                    
                m_copy["match_score"] = pct
                temp_list.append(m_copy)
                
            # Lấy Top K phim
            final_recs = sorted(temp_list, key=lambda x: x.get("match_score", 0.0), reverse=True)[:request.top_k]
            return RecommendResponse(recommendations=final_recs)

        # FALLBACK KHI KHÔNG ĐĂNG NHẬP
        fallback_recs = get_recommender().recommend(
            user_id=request.user_id,
            top_k=request.top_k,
            session_context=request.session_context,
            exclude_seen=request.exclude_seen,
        )
        
        for rec in fallback_recs:
            m_id = int(float(rec.get("movie_id", rec.get("movieId", rec.get("id", 0)))))
            if m_id in movie_truth:
                m_copy = movie_truth[m_id].copy()
                ai_score = float(rec.get("score", 0.0))
                m_copy["match_score"] = max(0.1, min(1.0, ai_score / 5.0)) if ai_score > 0 else 0.85
                final_recs.append(m_copy)
            else:
                final_recs.append(rec)

        return RecommendResponse(recommendations=final_recs)
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc