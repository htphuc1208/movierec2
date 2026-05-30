from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# ==========================================
# 1. HÀM GỌI API
# ==========================================
def api_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=2.0)
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

# ==========================================
# 2. HÀM TẢI DỮ LIỆU
# ==========================================
def load_users() -> list[int]:
    data = api_get("/users")
    return data["users"] if data and "users" in data else []

def load_filters() -> tuple[list[str], list[int]]:
    data = api_get("/filters")
    if data:
        return ["All"] + data.get("genres", []), ["All"] + data.get("years", [])
    return ["All"], ["All"]

def load_movies(genre: str = "All", year: str = "All", search: str = "") -> list[dict[str, Any]]:
    params = {}
    if genre != "All": params["genre"] = genre
    if year != "All": params["year"] = year
    if search: params["search"] = search
    data = api_get("/movies", params=params)
    return data["movies"] if data and "movies" in data else []

def load_movie_details(movie_id: str) -> dict[str, Any]:
    data = api_get(f"/movies/{movie_id}")
    return data if data else {}

def load_user_history(user_id: int) -> list[dict[str, Any]]:
    data = api_get(f"/users/{user_id}/history")
    return data["history"] if data and "history" in data else []

def load_similar_movies(movie_id: str) -> list[dict[str, Any]]:
    data = api_get(f"/movies/{movie_id}/similar")
    return data["similar"] if data and "similar" in data else []

def recommend(user_id: int, top_k: int, model_type: str, w_cf: float, w_cb: float, exclude_seen: bool) -> list[dict[str, Any]]:
    payload = {
        "user_id": user_id, "top_k": top_k, "model_type": model_type,
        "w_cf": w_cf, "w_cb": w_cb, "exclude_seen": exclude_seen,
    }
    data = api_post("/recommend", payload)
    return data["recommendations"] if data and "recommendations" in data else []

# ==========================================
# 3. QUẢN LÝ TRẠNG THÁI ĐIỀU HƯỚNG (ROUTING)
# ==========================================
def view_movie_detail(movie_id: str):
    st.session_state.viewing_movie_id = movie_id

def go_home():
    st.session_state.viewing_movie_id = None

# ==========================================
# 4. COMPONENT GIAO DIỆN
# ==========================================
def render_card(movie: dict[str, Any]) -> None:
    """Hiển thị thẻ phim dạng Grid ở trang chủ."""
    movie_id = str(movie.get("movieId", ""))
    title = movie.get("title", "Untitled")
    score = movie.get("score")
    genres = str(movie.get("genres", "")).replace("|", ", ")
    poster_url = str(movie.get("poster_url", ""))
    
    with st.container(border=True):
        if poster_url and poster_url.startswith("http"):
            st.image(poster_url, use_container_width=True)
        else:
            st.info("Chưa có ảnh Poster")
            
        st.markdown(f"**{title}**")
        if score is not None:
            st.caption(f"Độ phù hợp: {float(score):.2f} | {genres}")
        else:
            st.caption(genres)
        
        st.button("Xem chi tiết", key=f"btn_detail_{movie_id}", on_click=view_movie_detail, args=(movie_id,), use_container_width=True)

def render_history_card(movie: dict[str, Any]) -> None:
    """Hiển thị thẻ phim dạng Ngang (Horizontal) cực gọn cho Sidebar."""
    movie_id = str(movie.get("movieId", ""))
    title = movie.get("title", "Untitled")
    score = movie.get("score") # Ở lịch sử thì đây là điểm user đã đánh giá
    genres = str(movie.get("genres", "")).replace("|", ", ")
    poster_url = str(movie.get("poster_url", ""))
    
    with st.container(border=True):
        # Chia thẻ làm 2 cột: Cột trái (1 phần) cho Ảnh, Cột phải (2.5 phần) cho Chữ
        col1, col2 = st.columns([1, 2.5])
        
        with col1:
            if poster_url and poster_url.startswith("http"):
                st.image(poster_url, use_container_width=True)
            else:
                st.caption("No img")
                
        with col2:
            st.markdown(f"**{title}**")
            if score is not None:
                # Hiện điểm đánh giá
                st.markdown(f"⭐ **{score}**")
            st.caption(genres)
            # Nút bấm gọn gàng
            st.button("Chi tiết", key=f"btn_hist_{movie_id}", on_click=view_movie_detail, args=(movie_id,), use_container_width=True)

def render_detail_page():
    """Hiển thị toàn bộ thông tin chi tiết khi click vào 1 bộ phim."""
    movie_id = st.session_state.viewing_movie_id
    movie = load_movie_details(movie_id)
    
    # Thanh điều hướng
    st.button("Quay lại Trang chủ", on_click=go_home, type="primary")
    st.divider()

    if not movie:
        st.error("Không thể tải thông tin bộ phim này.")
        return

    # Layout chi tiết phim (2 cột)
    col1, col2 = st.columns([1, 2.5])
    with col1:
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
        else:
            st.info("No Poster Available")
            
    with col2:
        st.title(movie.get("title", "Untitled"))

        genres_display = str(movie.get('genres', 'N/A')).replace('|', ', ')
        cast_display = str(movie.get('cast', 'Đang cập nhật')).replace('|', ', ')
        
        # Nhóm metadata
        st.markdown(f"**Thể loại:** {genres_display}")
        st.markdown(f"**Diễn viên:** {cast_display}")
        st.markdown(f"**Đạo diễn:** {movie.get('director', 'Đang cập nhật')}")
        st.markdown(f"**Năm phát hành:** {movie.get('year', 'N/A')}")
        st.markdown(f"**Quốc gia:** {movie.get('country', 'Đang cập nhật')}")

        
        st.subheader("Nội dung chính")
        st.write(movie.get("overview", "Chưa có thông tin tóm tắt."))

    st.divider()
    
    # Gợi ý phim tương đồng (Two-Tower)
    st.subheader("Có thể bạn cũng thích (Phim tương tự)")
    similar_movies = load_similar_movies(movie_id)
    
    if similar_movies:
        for row_start in range(0, min(len(similar_movies), 10), 5):
            cols = st.columns(5)
            for col, sim_movie in zip(cols, similar_movies[row_start : row_start + 5]):
                with col:
                    render_card(sim_movie)
    else:
        st.info("Hiện chưa có gợi ý tương đồng cho bộ phim này.")

def render_home_page():
    """Hiển thị trang chủ mặc định (Đề xuất & Khám phá)."""
    users = load_users()
    
    # --- SIDEBAR TÀI KHOẢN ---
    with st.sidebar:
        st.header("Tài khoản")
        user_options = ["Guest"] + [str(u) for u in users]
        selected_user = st.selectbox("Đăng nhập dưới quyền User:", user_options)
        user_id = None if selected_user == "Guest" else int(selected_user)
            
        st.divider()
        if user_id:
            st.subheader(f"Lịch sử User {user_id}")
            history = load_user_history(user_id)
                
            if history:
                for movie in history:
                        render_history_card(movie)
            else:
                    st.caption("Chưa có lịch sử tương tác.")
        else:
            st.info("Đăng nhập để xem lịch sử tương tác.")

    st.title("Kho Phim Của Bạn")
    view_mode = st.radio("Chế độ xem:", ["Đề xuất cho bạn", "Lọc theo..."], horizontal=True, label_visibility="collapsed")
    st.divider()

    # --- LUỒNG 1: ĐỀ XUẤT ---
    if view_mode == "Đề xuất cho bạn":
        if not user_id:
            st.warning("Bạn đang ở chế độ Khách (Guest). Hãy đăng nhập ở cột trái để nhận đề xuất cá nhân hóa.")
        else:
            with st.expander("Tinh chỉnh Thuật toán Gợi ý (Dành cho Dev)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    model_type = st.selectbox("Chọn mô hình:", ["hybrid", "svd", "lightgcn", "twotower"])
                    top_k = st.slider("Số lượng gợi ý:", 5, 20, 10, 1)
                with col2:
                    w_cf = st.slider("Trọng số LightGCN (w_cf):", 0.0, 1.0, 0.5, 0.1)
                    w_cb = st.slider("Trọng số Two-Tower (w_cb):", 0.0, 1.0, round(1.0 - w_cf, 1), 0.1)
                exclude_seen = st.toggle("Ẩn phim đã xem", value=True)

            if st.button("Làm mới Đề xuất", type="primary"):
                with st.spinner(f"Đang tính toán ma trận ({model_type.upper()})..."):
                    recommendations = recommend(user_id, top_k, model_type, w_cf, w_cb, exclude_seen)
                    if recommendations:
                        for row_start in range(0, len(recommendations), 4):
                            cols = st.columns(4)
                            for col, movie in zip(cols, recommendations[row_start : row_start + 4]):
                                with col:
                                    render_card(movie)
                    else:
                        st.info("Không tìm thấy đề xuất phù hợp.")

    # --- LUỒNG 2: LỌC & TÌM KIẾM ---
    elif view_mode == "Lọc theo...":
        genres_opt, years_opt = load_filters()
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            search_query = st.text_input("Tên phim:")
        with f_col2:
            sel_genre = st.selectbox("Thể loại:", genres_opt)
        with f_col3:
            sel_year = st.selectbox("Năm phát hành:", years_opt)
            
        filtered_movies = load_movies(genre=sel_genre, year=str(sel_year), search=search_query)
        st.write(f"Tìm thấy **{len(filtered_movies)}** bộ phim phù hợp.")
        
        for row_start in range(0, min(len(filtered_movies), 20), 4):
            cols = st.columns(4)
            for col, movie in zip(cols, filtered_movies[row_start : row_start + 4]):
                with col:
                    render_card(movie)

# ==========================================
# 5. MAIN APP ROUTER
# ==========================================
def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")
    
    # Khởi tạo biến lưu trạng thái chuyển trang
    if "viewing_movie_id" not in st.session_state:
        st.session_state.viewing_movie_id = None

    # Logic điều hướng: Có ID thì mở Trang chi tiết, Không thì mở Trang chủ
    if st.session_state.viewing_movie_id:
        render_detail_page()
    else:
        render_home_page()

if __name__ == "__main__":
    main()