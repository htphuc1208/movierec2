from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import MovieLensDataLoader
from models import HybridMovieRecommender

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

def clean_title(raw_title: str) -> str:
    """Định dạng lại tên phim: chuyển mạo từ (The, A, An) lên đầu."""
    if not isinstance(raw_title, str):
        return "Untitled"
    cleaned = re.sub(r'^(.*?),\s*(The|A|An)\s*(\(\d{4}\))$', r'\2 \1 \3', raw_title, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(.*?),\s*(The|A|An)$', r'\2 \1', cleaned, flags=re.IGNORECASE)
    return cleaned

# ==========================================
# 1. GỌI API & DATA
# ==========================================
@lru_cache(maxsize=1)
def local_recommender() -> HybridMovieRecommender:
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
        except Exception:
            pass
    model.fit(bundle.movies, bundle.ratings, bundle.tags)
    model.dataset_name = data_dir
    return model

def api_get(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=1.5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def load_movies() -> list[dict[str, Any]]:
    data = api_get("/movies")
    if data and "movies" in data:
        return data["movies"]
    return local_recommender().movies_for_picker()

def load_users() -> list[int]:
    data = api_get("/users")
    if data and "users" in data:
        return data["users"]
    return local_recommender().users()

def recommend(user_id: int | None, top_k: int = 15, session_context: list = None, exclude_seen: bool = True, model_name: str = "SVD"):
    url = f"{API_URL}/recommend"
    payload = {
        "user_id": user_id,
        "top_k": top_k,
        "session_context": session_context or [],
        "exclude_seen": exclude_seen,
        "model_name": model_name
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("recommendations", [])
        return []
    except Exception as e:
        print(f"Lỗi gọi API recommend: {e}")
        return []

# ==========================================
# 2. ĐIỀU HƯỚNG & SESSION STATE
# ==========================================
def navigate(page: str, movie_id: str | None = None):
    st.session_state.current_page = page
    if movie_id:
        st.session_state.selected_movie_id = movie_id

def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"
    if "selected_movie_id" not in st.session_state:
        st.session_state.selected_movie_id = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

# ==========================================
# 3. COMPONENT GIAO DIỆN
# ==========================================
def handle_search():
    query = st.session_state.search_box_widget
    
    if query.strip():
        st.session_state.search_query = query
        st.session_state.current_page = "search"
        
        st.session_state.search_box_widget = ""
        
def clear_search_and_home():
    st.session_state.search_query = ""
    st.session_state.search_box_widget = ""
    navigate("home")

def render_navbar(users: list[int]):
    col1, col2, col3, col4 = st.columns([1.5, 4, 1.5, 1], vertical_alignment="center")
    
    with col1:
        st.button("**movierec**", on_click=clear_search_and_home, width="stretch")
            
    with col2:
        st.text_input(
            "Tìm kiếm",
            placeholder="Gõ tên phim rồi Enter (ví dụ: Toy Story)...",
            label_visibility="collapsed",
            key="search_box_widget",
            on_change=handle_search 
        )
            
    with col3:
        user_options = ["Guest"] + [str(user) for user in users]
        selected_user = st.selectbox("Tài khoản", user_options, label_visibility="collapsed")
        st.session_state.current_user = selected_user
        
    with col4:
        def go_to_history():
            st.session_state.current_page = "history"
            
        if selected_user != "Guest":
            st.button("Lịch sử", on_click=go_to_history, width="stretch")
        else:
            st.button("Lịch sử", disabled=True, width="stretch", help="Vui lòng chọn User để xem lịch sử")
            
    st.divider()

def render_movie_card(movie: dict[str, Any], idx: int):
    raw_id = movie.get("movieId", movie.get("movie_id", movie.get("id")))
    movie_id = str(raw_id) if raw_id else ""
    
    poster_url = movie.get("poster_url") or ""
    title = clean_title(movie.get("title", "Untitled"))

    score = float(movie.get("score", 0.0))
    vote_count = int(movie.get("vote_count", 0))
    match_score = movie.get("match_score")
    genres = str(movie.get("genres", "")).replace("|", ", ")

    if score > 5.0:
        score = score / 2.0
        
    rating_str = f"⭐{score:.1f} ({vote_count})" if score > 0 else "Chưa có điểm"

    if match_score is not None and 0.0 < match_score <= 1.0:
        match_html = f'<span style="color: #46d369; font-weight: bold;">{int(match_score * 100)}% Match</span>'
        display_html = f"{rating_str} • {genres}<br>{match_html}"
    else:
        display_html = f"{rating_str} • {genres}"

    with st.container(border=True):
        st.markdown('<span class="movie-card-marker" style="display: none;"></span>', unsafe_allow_html=True)
        
        if poster_url and str(poster_url).startswith("http"):
            st.markdown(f'''
                <div style="width: 100%; aspect-ratio: 2/3; border-radius: 8px; overflow: hidden; margin-bottom: 10px;">
                    <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
                <div style="width: 100%; aspect-ratio: 2/3; background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; color: #666; text-align: center;">
                    <div>Chưa có ảnh</div>
                </div>
            ''', unsafe_allow_html=True)
            
        st.markdown(f'''
            <div style="font-weight: bold; font-size: 15px; margin-bottom: 5px;
                        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; height: 45px;">
                {title}
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div style="font-size: 13px; color: #aaa; margin-bottom: 15px;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {display_html}
            </div>
        ''', unsafe_allow_html=True)
        
        if movie_id:
            st.button("Chi tiết", key=f"btn_{movie_id}_{idx}", on_click=navigate, args=("detail", movie_id), width="stretch")
        else:
            st.button("Lỗi ID", key=f"btn_err_{idx}", disabled=True, width="stretch")

def render_movie_row(title: str, movies: list[dict[str, Any]], row_id: int):
    if not movies:
        return
        
    st.markdown(f"##### {title}")
    
    cols = st.columns(len(movies))
    for idx, movie in enumerate(movies):
        with cols[idx]:
            render_movie_card(movie, row_id * 1000 + idx)
            
    st.markdown("<br>", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_cached_trending_hero():
    return api_get("/movies/trending?top_k=5")

def render_hero_banner():
    trending_data = get_cached_trending_hero()
    if not trending_data or not trending_data.get("movies"):
        return
        
    movies = trending_data["movies"]
    
    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker) {
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    scroll-snap-type: x mandatory;
    padding-bottom: 30px !important;
    scroll-behavior: smooth;
}
div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker) > div {
    min-width: 100% !important;
    flex: 0 0 100% !important;
    scroll-snap-align: start;
    background-color: transparent !important;
}
div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker)::-webkit-scrollbar { height: 10px; }
div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker)::-webkit-scrollbar-thumb { background: #E50914; border-radius: 10px; }
div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker)::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }

/* Ép nút bấm nổi lên trên ảnh nền */
div.stButton > button {
    position: relative !important;
    z-index: 10 !important;
}
</style>
""", unsafe_allow_html=True)
    
    # Dùng cột Streamlit làm khung trượt
    cols = st.columns(len(movies))
    
    for idx, col in enumerate(cols):
        with col:
            st.markdown('<span class="hero-carousel-marker" style="display:none;"></span>', unsafe_allow_html=True)
            
            m = movies[idx]
            raw_id = m.get("movie_id", m.get("movieId", m.get("id")))
            movie_id = str(int(float(raw_id))) if raw_id else ""
            
            title = m.get("title", "Untitled")
            score = float(m.get("score", 0.0))
            if score > 5.0: score /= 2.0
            genres = str(m.get("genres", "Hấp dẫn")).replace("|", ", ")
            
            poster = m.get("poster_url", "")
            if not poster or str(poster).lower() == "nan":
                poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"
                
            # Vẽ Background giao diện
            html_bg = f"""
<div style="width: 100%; min-height: 480px; border-radius: 12px; border: 1px solid #222; box-shadow: 0 8px 20px rgba(0,0,0,0.8); display: flex; flex-direction: column; justify-content: flex-end; padding: 45px; padding-bottom: 100px; box-sizing: border-box; background: linear-gradient(to right, rgba(15,15,15,1) 0%, rgba(15,15,15,0.7) 45%, rgba(15,15,15,0) 100%), linear-gradient(to top, rgba(15,15,15,1) 0%, rgba(15,15,15,0.5) 30%, rgba(15,15,15,0) 100%), url('{poster}') no-repeat center top / cover; background-color: #0f0f0f; margin-bottom: -80px; position: relative; z-index: 0;">
<div style="max-width: 65%;">
<h1 style="font-size: 3.6rem; margin: 0 0 10px 0; color: white; text-shadow: 2px 2px 5px rgba(0,0,0,1); line-height: 1.15; font-weight: 800; letter-spacing: -1px;">{title}</h1>
<div style="font-size: 1.15rem; color: #ddd; margin-bottom: 0px; display: flex; align-items: center; gap: 10px; text-shadow: 1px 1px 3px rgba(0,0,0,1);">
<span style="color: #F5C518; font-weight: bold;">⭐{score:.1f}</span>
<span style="color: #777;">|</span>
<span style="font-weight: 500;">{genres}</span>
</div>
</div>
</div>
"""
            st.markdown(html_bg, unsafe_allow_html=True)
            
            # Nút bấm tương tác
            btn_space, btn_xem, padding_right = st.columns([0.06, 0.2, 0.74])
            
            with btn_xem:
                if movie_id:
                    st.button("Xem chi tiết", type="primary", key=f"hero_det_{idx}_{movie_id}", on_click=navigate, args=("detail", movie_id), width="stretch")
                else:
                    st.button("Xem chi tiết", type="primary", key=f"hero_err_{idx}", disabled=True, width="stretch")
                    


def render_home_page(movies_df: pd.DataFrame = None):
    """Giao diện Trang chủ: Đa dạng các luồng phim chuẩn IMDb."""
    
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:has(.movie-card-marker) {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 20px !important;
        scroll-behavior: smooth;
    }
    div[data-testid="stHorizontalBlock"]:has(.movie-card-marker) > div {
        min-width: 230px !important; max-width: 230px !important; width: 230px !important;
        flex: 0 0 230px !important; margin-right: 15px !important;
    }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar { height: 10px; }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb { background: #f5c518; border-radius: 10px; }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    # 1. LẤY CHUẨN USER_ID TỪ ĐẦU
    selected_user = st.session_state.get("current_user", "Guest")
    user_id = None
    if selected_user != "Guest":
        try:
            # Ép kiểu an toàn, nếu là chuỗi "User 10" hay số "10" đều xử lý được
            user_id = int(str(selected_user).replace("User", "").strip())
        except ValueError:
            user_id = None

    render_hero_banner()
    
    # ---------------------------------------------------------
    # 3. HÀNG ĐỀ XUẤT CHO BẠN (TÍCH HỢP NÚT A/B TESTING)
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_title, col_model = st.columns([1, 1])
    with col_title:
        st.markdown("""
    <style>
    div[data-testid="stRadio"] { margin-bottom: -45px !important; }
    div[data-testid="stMarkdownContainer"] > h2,
    div[data-testid="stMarkdownContainer"] > h3,
    div[data-testid="stMarkdownContainer"] > h4,
    div[data-testid="stMarkdownContainer"] > h5 {
        margin-bottom: -25px !important;
        padding-bottom: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)
        #st.markdown("##### Được đề xuất cho bạn")
        
    with col_model:
        model_choice = st.radio(
            "Chọn Động cơ AI:", 
            ("LightGCN (Đồ thị Học sâu)", "SVD (Ma trận Truyền thống)"), 
            horizontal=True,
            label_visibility="collapsed"
        )
    
    actual_model_name = "LightGCN" if "LightGCN" in model_choice else "SVD"

    recs = recommend(
        user_id=user_id, 
        top_k=15, 
        session_context=[], 
        exclude_seen=True, 
        model_name=actual_model_name 
    )
    
    if recs:
        render_movie_row("Được đề xuất cho bạn", recs, row_id=1)
    else:
        st.info("Hãy chọn một User trên thanh Menu để AI bắt đầu gợi ý phim nhé!")

    # ---------------------------------------------------------
    # CÁC HÀNG PHIM BÊN DƯỚI
    # ---------------------------------------------------------
    latest_data = api_get("/movies/latest?top_k=15")
    if latest_data and "movies" in latest_data:
        render_movie_row("Phim Mới Ra Mắt", latest_data["movies"], row_id=99)

    trending_data = api_get("/movies/trending?top_k=15")
    if trending_data and "movies" in trending_data:
        render_movie_row("Đang Thịnh Hành", trending_data["movies"], row_id=2)
        
    top_data = api_get("/movies/top-rated?top_k=15")
    if top_data and "movies" in top_data:
        render_movie_row("Top Phim Đánh Giá Cao Nhất", top_data["movies"], row_id=3)
        
    action_data = api_get("/movies/genre/Action?top_k=15")
    if action_data and "movies" in action_data:
        render_movie_row("Phim Hành Động Kịch Tính", action_data["movies"], row_id=4)

    comedy_data = api_get("/movies/genre/Comedy?top_k=15")
    if comedy_data and "movies" in comedy_data:
        render_movie_row("Phim Hài Hước Giải Trí", comedy_data["movies"], row_id=5)

def render_search_page():
    query = st.session_state.get("search_query", "")
    st.button("Trở về trang chủ", on_click=navigate, args=("home",))
    st.subheader(f"Kết quả tìm kiếm cho: '{query}'")
    
    data = api_get(f"/movies?search={query}")
    results = data.get("movies", []) if data else []
    
    if not results:
        st.warning("Không tìm thấy bộ phim nào phù hợp.")
        return
        
    cols = st.columns(5)
    for idx, movie in enumerate(results[:20]):
        with cols[idx % 5]:
            render_movie_card(movie, idx)

def render_history_page():
    st.markdown("<h2 style='color: white; font-weight: 800;'>Lịch sử xem phim của bạn</h2>", unsafe_allow_html=True)
    
    if st.button("Quay lại Trang chủ", width="content"):
        navigate("home")
        
    st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
        
    selected_user = st.session_state.get("current_user", "Guest")
    if selected_user == "Guest":
        st.warning("Vui lòng đăng nhập / chọn User ở thanh menu để xem lịch sử.")
        return
        
    try:
        user_id = int(str(selected_user).replace("User", "").strip())
    except ValueError:
        return
        
    col_sort, _ = st.columns([1, 3])
    with col_sort:
        sort_option = st.selectbox(
            "Sắp xếp theo:", 
            ["Gần đây / Tương tác cao", "Điểm đánh giá: Giảm dần", "Tên phim: A -> Z", "Tên phim: Z -> A"]
        )

    with st.spinner("Đang tải dữ liệu lịch sử..."):
        history_data = api_get(f"/users/{user_id}/history")
        movies = history_data.get("movies", []) if history_data else []
        
    if not movies:
        st.info("Bạn chưa có lịch sử xem phim hoặc đánh giá nào.")
        return

    if sort_option == "Điểm đánh giá: Giảm dần":
        movies = sorted(movies, key=lambda x: x.get("user_rating", 0.0), reverse=True)
    elif sort_option == "Tên phim: A -> Z":
        movies = sorted(movies, key=lambda x: str(x.get("title", "")).lower())
    elif sort_option == "Tên phim: Z -> A":
        movies = sorted(movies, key=lambda x: str(x.get("title", "")).lower(), reverse=True)

    st.markdown("<br>", unsafe_allow_html=True)
    num_cols = 5
    
    for i in range(0, len(movies), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(movies):
                m = movies[i + j]
                title = m.get("title", "Untitled")
                poster = m.get("poster_url", "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80")
                rating = m.get("user_rating", "Chưa chấm")
                raw_id = m.get("movie_id", m.get("movieId", m.get("id")))
                movie_id = str(int(float(raw_id))) if raw_id else ""
                
                with cols[j]:
                    st.markdown(f"""
                    <div style="background-color: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 12px; margin-bottom: 10px; transition: 0.3s;">
                        <img src="{poster}" style="width: 100%; border-radius: 6px; aspect-ratio: 2/3; object-fit: cover;">
                        <h5 style="color: white; margin-top: 12px; margin-bottom: 5px; font-size: 1rem; text-overflow: ellipsis; white-space: nowrap; overflow: hidden;" title="{title}">{title}</h5>
                        <p style="color: #f5c518; font-weight: bold; margin: 0; font-size: 0.9rem;">⭐Điểm đánh giá: {rating}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if movie_id:
                        st.button("Xem chi tiết", key=f"hist_btn_{movie_id}_{i}_{j}", on_click=navigate, args=("detail", movie_id), width="stretch")

def render_detail_page(movie_id):
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:has(.movie-card-marker) {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 20px !important;
        scroll-behavior: smooth;
    }
    div[data-testid="stHorizontalBlock"]:has(.movie-card-marker) > div {
        min-width: 230px !important; max-width: 230px !important; width: 230px !important;
        flex: 0 0 230px !important; margin-right: 15px !important;
    }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar { height: 10px; }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb { background: #f5c518; border-radius: 10px; }
    div[data-testid="stHorizontalBlock"]::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)
    
    st.button("Quay lại Trang chủ", on_click=navigate, args=("home",), type="secondary")
    
    movie = api_get(f"/movies/{movie_id}")
    
    if not movie:
        st.warning("Không tìm thấy thông tin chi tiết bộ phim này.")
        return

    title = movie.get("title", "Untitled")
    poster_url = movie.get("poster_url", "")
    genres = str(movie.get("genres", "")).replace("|", ", ")
    score = float(movie.get("score", 0.0))
    vote_count = int(movie.get("vote_count", 0))
    if score > 5.0: score = score / 2.0

    overview = movie.get("overview", "Chưa có thông tin tóm tắt cho bộ phim này.")
    director = movie.get("director", "Đang cập nhật")
    cast = str(movie.get("cast", "Đang cập nhật")).replace("|", ", ")
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.5], gap="large")
    
    with c1:
        if poster_url:
            st.image(poster_url, width="stretch", clamp=True)
            
    with c2:
        st.markdown(f"<h1 style='font-size: 3rem; margin-bottom: 0;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #F5C518; font-size: 1.3rem; font-weight: bold;'>⭐{score:.1f} <span style='color: #aaa; font-weight: normal; font-size: 1rem;'>({vote_count} đánh giá)</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Đạo diễn:</b> {director}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Diễn viên:</b> {cast}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Thể loại:</b> {genres}</p>", unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: #aaa; line-height: 1.6; margin-top: 15px;'>{overview}</p>", unsafe_allow_html=True)
        
        # ===============================
        # KHOANG ĐÁNH GIÁ PHIM (RATING)
        # ===============================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 10px;'>Đánh giá của bạn</h4>", unsafe_allow_html=True)
        
        # 1. Trích xuất chuẩn user_id từ biến session
        selected_user = st.session_state.get("current_user", "Guest")
        user_id = None
        if selected_user != "Guest":
            try:
                user_id = int(str(selected_user).replace("User", "").strip())
            except ValueError:
                pass
                
        # 2. Logic hiển thị thanh đánh giá chặn Reload & Chặn đánh giá lại
        if not user_id:
            st.info("Vui lòng chọn User trên thanh Menu để đánh giá phim này.")
        else:
            # Gọi API kiểm tra lịch sử rate
            existing_rating_data = api_get(f"/rate/{user_id}/{movie_id}")
            existing_rating = existing_rating_data.get("rating") if existing_rating_data else None
            
            rating_options = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
            
            with st.form(key=f"form_rating_{movie_id}", border=False):
                col_rate, col_btn, _ = st.columns([1.2, 1.2, 4], vertical_alignment="center")
                
                with col_rate:
                    if existing_rating is not None and existing_rating in rating_options:
                        default_idx = rating_options.index(existing_rating)
                        user_rating = st.selectbox(
                            "Chấm điểm", 
                            options=rating_options,
                            index=default_idx,
                            format_func=lambda x: f"{x} ⭐",
                            label_visibility="collapsed",
                        )
                    else:
                        # Chưa đánh giá: Mở Menu bình thường
                        user_rating = st.selectbox(
                            "Chấm điểm", 
                            options=rating_options,
                            format_func=lambda x: f"{x} ⭐",
                            label_visibility="collapsed"
                        )
                
                with col_btn:
                    # Đổi trạng thái Nút bấm tùy theo việc đã rate hay chưa
                    if existing_rating is not None:
                        submit_btn = st.form_submit_button("Đã đánh giá", type="secondary", disabled=True, width="stretch")
                    else:
                        submit_btn = st.form_submit_button("Gửi đánh giá", type="primary", width="stretch")
                    
                # Chỉ xử lý gửi data khi nút Submit được bấm
                if submit_btn and existing_rating is None:
                    payload = {
                        "user_id": user_id,
                        "movie_id": int(float(movie_id)),
                        "rating": float(user_rating)
                    }
                    response = api_post("/rate", payload)
                    
                    if response and response.get("status") == "success":
                        st.toast(f"Đã ghi nhận đánh giá {user_rating} ⭐ thành công!")
                        # Tải lại trang ngay lập tức để Khóa form
                        st.rerun()
                    else:
                        st.toast("Error: Có lỗi xảy ra, chưa thể gửi đánh giá.")
        # ====================================

    st.markdown("<hr style='border-color: #333; margin: 3rem 0 1rem 0;'>", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-bottom: 0;'>Có thể bạn cũng thích</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #aaa; font-size: 0.9rem; margin-bottom: 20px;'><i>Gợi ý dựa trên nội dung và thể loại</i></p>", unsafe_allow_html=True)
    
    similar_data = api_get(f"/movies/{movie_id}/similar?top_k=15")
    
    if similar_data and similar_data.get("movies"):
        filtered_similar = [
            m for m in similar_data["movies"] 
            if str(int(float(m.get("movie_id", m.get("movieId", m.get("id")))))) != str(movie_id)
        ]
        
        render_movie_row("", filtered_similar, row_id=9999)
    else:
        st.info("Hệ thống đang cập nhật thêm phim tương tự...")

# ==========================================
# 4. HÀM MAIN CHẠY ỨNG DỤNG
# ==========================================
def main() -> None:
    st.set_page_config(page_title="movierec", layout="wide")
    
    init_session_state()
    
    movies = load_movies()
    users = load_users()
    movie_frame = pd.DataFrame(movies)
    
    render_navbar(users)
    
    if st.session_state.current_page == "home":
        render_home_page(movie_frame)
    elif st.session_state.current_page == "detail":
        current_movie = st.session_state.get("selected_movie_id") 
        render_detail_page(current_movie)
    elif st.session_state.current_page == "search":
        render_search_page()
    elif st.session_state.current_page == "history":
        render_history_page()

if __name__ == "__main__":
    main()