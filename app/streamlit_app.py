"""Streamlit frontend for the movie recommender."""

from __future__ import annotations

import html
import os
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

http_session = requests.Session()

retries = Retry(total=3, backoff_factor=0.1)
http_session.mount('http://', HTTPAdapter(max_retries=retries))

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str, **params: Any) -> Any | None:
    try:
        response = http_session.get(f"{API_URL}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.toast(f"Không gọi được API: {exc}")
        return None
    
@st.cache_data(ttl=600, show_spinner=False)
def api_get_cached(path: str, **params: Any) -> Any | None:
    return api_get(path, **params)


def api_post(path: str, payload: dict[str, Any]) -> Any | None:
    try:
        response = http_session.post(f"{API_URL}{path}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.toast(f"Không gọi được API: {exc}")
        return None


def movie_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("movies") or data.get("recommendations") or []
    return []


def load_users() -> list[int]:
    data = api_get("/users")
    return data.get("users", []) if isinstance(data, dict) else []


def clean_title(title: str) -> str:
    raw_title = str(title or "Không rõ tên").strip()
    cleaned = re.sub(r"^(.*?),\s*(The|A|An)\s*(\(\d{4}\))$", r"\2 \1 \3", raw_title, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(.*?),\s*(The|A|An)$", r"\2 \1", cleaned, flags=re.IGNORECASE)
    return cleaned


def safe_text(value: Any) -> str:
    return html.escape(str(value or ""))


def parse_movie_id(movie: dict[str, Any]) -> int:
    raw_id = movie.get("movie_id") or movie.get("movieId") or movie.get("id") or 0
    try:
        return int(float(raw_id))
    except (TypeError, ValueError):
        return 0


def navigate(page: str, movie_id: int | None = None, tags: list[str] | None = None) -> None:
    st.session_state.page = page
    if movie_id is not None:
        st.session_state.movie_id = int(movie_id)
    st.session_state.current_tags = tags if tags is not None else []


def init_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("movie_id", None)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("current_tags", [])


def submit_search():
    query = st.session_state.search_input_widget
    if query.strip():
        st.session_state.search_query = query
        navigate("search")
        st.session_state.search_input_widget = ""


def render_navbar(users: list[int]) -> None:
    if "search_input_widget" not in st.session_state:
        st.session_state.search_input_widget = ""

    col_brand, col_search, col_user, col_history = st.columns([1.1, 3.5, 1.4, 1.1])
    
    with col_brand:
        if st.button("movierec", use_container_width=True):
            navigate("home")
            
    with col_search:
        st.text_input(
            "Tìm phim", 
            key="search_input_widget", 
            on_change=submit_search, 
            label_visibility="collapsed", 
            placeholder="Tìm phim"
        )
        
    with col_user:
        options = ["Guest"] + [str(user) for user in users[:1000]]
        current = "Guest" if st.session_state.current_user is None else str(st.session_state.current_user)
        selected = st.selectbox("User", options, index=options.index(current) if current in options else 0, label_visibility="collapsed")
        st.session_state.current_user = None if selected == "Guest" else int(selected)
        
    with col_history:
        st.button("Lịch sử", use_container_width=True, disabled=st.session_state.current_user is None, on_click=navigate, args=("history",))


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(.movie-row-marker) {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            padding-bottom: 18px !important;
            scroll-behavior: smooth;
        }
        div[data-testid="stHorizontalBlock"]:has(.movie-row-marker) > div {
            min-width: 220px !important;
            max-width: 220px !important;
            flex: 0 0 220px !important;
            margin-right: 12px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.movie-row-marker)::-webkit-scrollbar { height: 9px; }
        div[data-testid="stHorizontalBlock"]:has(.movie-row-marker)::-webkit-scrollbar-thumb { background: #f5c518; border-radius: 10px; }
        div[data-testid="stHorizontalBlock"]:has(.movie-row-marker)::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.08); border-radius: 10px; }

        
        button[data-baseweb="tab"] {
            font-size: 1.25rem !important; 
            font-weight: 600 !important;
            padding: 1rem 1.5rem !important;
            color: #888 !important; /* Màu xám nhạt cho Tab chưa chọn */
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #F5C518 !important; 
            font-size: 1.3rem !important;
        }
        
        /* Kéo giãn khoảng cách giữa các Tab và thêm đường kẻ mờ ở dưới đáy */
        div[data-baseweb="tab-list"] {
            gap: 15px;
            border-bottom: 2px solid #222 !important; 
            margin-bottom: 1.5rem;
        }
        
        div[data-baseweb="tab-highlight"] {
            background-color: #F5C518 !important;
            height: 4px !important;
            border-radius: 4px 4px 0 0 !important;
        }

        .block-container {
            padding-top: 1rem !important;
        }
        
        div[data-testid="stTabs"] > div:first-child {
            position: sticky !important;
            top: 0 !important;
            z-index: 9999 !important;
            padding-top: 20px !important;
            padding-bottom: 5px !important;
            box-shadow: 0px 10px 10px -10px rgba(0,0,0,0.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner() -> None:
    # 1. Gọi API lấy 5 phim Trending
    data = api_get_cached("/movies/trending", top_k=5)
    movies = movie_list(data)
    if not movies:
        return
        
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker) {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important; /* [MỚI] Triệt tiêu thanh trượt dọc */
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
            
            # Sử dụng các hàm parse chuẩn của phiên bản mới
            m = movies[idx]
            movie_id = parse_movie_id(m)
            
            title = safe_text(clean_title(m.get("title", "Untitled")))
            score = float(m.get("vote_average") or m.get("score") or 0.0)
            genres = safe_text(str(m.get("tmdb_genres") or m.get("genres") or "Hấp dẫn").replace("|", ", "))
            
            poster = m.get("poster_url") or ""
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
                if movie_id > 0:
                    st.button("Xem chi tiết", type="primary", key=f"hero_det_{idx}_{movie_id}", on_click=navigate, args=("detail", movie_id), width="stretch")
                else:
                    st.button("Xem chi tiết", type="primary", key=f"hero_err_{idx}", disabled=True, width="stretch")


def render_movie_card(movie: dict[str, Any], key: str, is_row: bool = False) -> None:
    movie_id = parse_movie_id(movie)
    
    poster_url = movie.get("poster_url") or ""
    title = clean_title(movie.get("title", "Untitled"))

    score = float(movie.get("vote_average") or movie.get("score") or 0.0)
    vote_count = int(movie.get("vote_count") or 0)
    match_score = movie.get("match_score")
    genres = str(movie.get("tmdb_genres") or movie.get("genres") or "").replace("|", ", ")
    
    explanation_tags = movie.get("explanation_tags") or []

    if score > 5.0:
        score = score / 2.0
        
    rating_str = f"⭐{score:.1f}" if score > 0 else "Chưa có điểm"

    if match_score is not None and 0.0 < match_score <= 1.0:
        match_html = f'<span style="color: #46d369; font-weight: bold; margin-right: 6px; font-size: 11px;">{int(match_score * 100)}% Match</span>'
    else:
        match_html = ''

    tags_html = ""
    if explanation_tags:
        for tag in explanation_tags[:2]:
            tags_html += f'<span style="background-color: rgba(245, 197, 24, 0.15); color: #F5C518; border: 1px solid rgba(245, 197, 24, 0.3); border-radius: 4px; padding: 2px 6px; font-size: 10px; margin-right: 4px; display: inline-block; white-space: nowrap;">{safe_text(tag)}</span>'

    if poster_url and str(poster_url).startswith("http"):
        img_html = f'<img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover;">'
    else:
        img_html = '<div style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; color: #666;">Chưa có ảnh</div>'

    marker_html = '<span class="movie-row-marker" style="display: none;"></span>' if is_row else ''

    with st.container(border=True):
        html_content = f"""{marker_html}
<div style="width: 100%; aspect-ratio: 2/3; background-color: rgba(255, 255, 255, 0.05); border-radius: 8px; overflow: hidden; margin-bottom: 12px;">
{img_html}
</div>
<div style="height: 105px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="font-weight: bold; font-size: 15px; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
{title}
</div>
<div style="font-size: 13px; color: #aaa; margin-top: auto;">
<div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
{rating_str} • {genres}
</div>
<div style="display: flex; flex-wrap: wrap; align-items: center; min-height: 24px; margin-top: 4px;">
{match_html}{tags_html}
</div>
</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)
        
        if movie_id > 0:
            st.button("Chi tiết", key=f"btn_{key}_{movie_id}", on_click=navigate, args=("detail", movie_id, explanation_tags), use_container_width=True)
        else:
            st.button("Lỗi ID", key=f"btn_err_{key}", disabled=True, use_container_width=True)

def render_movie_row(title: str, movies: list[dict[str, Any]], row_key: str) -> None:
    if not movies:
        return
    if title.strip():
        st.markdown(f"<h3 style='margin-top: 15px;'>{title}</h3>", unsafe_allow_html=True)
    
    columns = st.columns(min(15, len(movies)))
        
    for idx, movie in enumerate(movies[:15]):
        with columns[idx % len(columns)]:
            render_movie_card(movie, f"{row_key}_{idx}", is_row=True)


def recommend(user_id: int | None, model_name: str, top_k: int = 15) -> list[dict[str, Any]]:
    data = api_post(
        "/recommend",
        {
            "user_id": user_id,
            "top_k": top_k,
            "session_context": [],
            "exclude_seen": True,
            "model_name": model_name,
        },
    )
    return movie_list(data)


def render_home_page() -> None:
    render_hero_banner()
    
    user_id = st.session_state.current_user
    
    if user_id is not None:
        col_title, col_model = st.columns([5, 1], vertical_alignment="bottom")
        
        with col_title:
            st.markdown("<h3 style='margin-top: 15px; margin-bottom: 0;'>Được đề xuất cho bạn</h3>", unsafe_allow_html=True)
            
        with col_model:
            model_label = st.selectbox(
                "Chọn mô hình", 
                ["Hybrid", "LightGCN", "Content", "Popularity"],
                format_func=lambda x: f"Mô hình: {x}", 
                help="Chọn thuật toán AI để hệ thống tính toán và đưa ra danh sách phim phù hợp nhất với bạn.",
                label_visibility="collapsed"
            )
        
        recs = recommend(user_id, model_name=str(model_label).lower(), top_k=15)
        
        render_movie_row("", recs, "recommended")
        
    else:
        st.info("Chọn user để xem gợi ý cá nhân hóa.")

    for title, path in [
        ("Phim mới", "/movies/latest"),
        ("Đang thịnh hành", "/movies/trending"),
        ("Đánh giá cao", "/movies/top-rated"),
        ("Hành động", "/movies/genre/Action"),
        ("Hài hước", "/movies/genre/Comedy"),
    ]:
        render_movie_row(title, movie_list(api_get_cached(path, top_k=15)), path)


def render_search_page() -> None:
    query = st.session_state.search_query.strip()
    st.button("Về trang chủ", on_click=navigate, args=("home",))
    st.title(f"Kết quả tìm kiếm: {query}")
    results = movie_list(api_get("/movies", query=query, limit=50))
    if not results:
        st.warning("Không tìm thấy phim phù hợp.")
        return
    for start in range(0, len(results[:30]), 5):
        columns = st.columns(5)
        for idx, movie in enumerate(results[start : start + 5]):
            with columns[idx]:
                render_movie_card(movie, f"search_{start}_{idx}")


def render_history_page() -> None:
    st.button("Về trang chủ", on_click=navigate, args=("home",))
    st.title("Lịch sử của bạn")
    
    user_id = st.session_state.current_user
    if user_id is None:
        st.warning("Chọn user để xem lịch sử.")
        return
        
    movies = movie_list(api_get(f"/users/{user_id}/history", top_k=50))
    if not movies:
        st.info("Chưa có lịch sử trong artifacts hoặc rating sidecar.")
        return
        
    sort_option = st.selectbox("Sắp xếp", ["Nguồn dữ liệu", "Điểm đánh giá giảm dần", "Tên phim A-Z", "Tên phim Z-A"])
    
    if sort_option == "Điểm đánh giá giảm dần":
        movies = sorted(movies, key=lambda movie: float(movie.get("user_rating") or 0.0), reverse=True)
    elif sort_option == "Tên phim A-Z":
        movies = sorted(movies, key=lambda movie: clean_title(movie.get("title", "")).lower())
    elif sort_option == "Tên phim Z-A":
        movies = sorted(movies, key=lambda movie: clean_title(movie.get("title", "")).lower(), reverse=True)
        
    rows = []
    for movie in movies:
        rows.append(
            {
                "movie_id": movie.get("movie_id"),
                "title": clean_title(movie.get("title", "")),
                "user_rating": movie.get("user_rating"),
                "source": movie.get("history_source"),
            }
        )

    
        
    with st.expander("Xem chi tiết dữ liệu thô"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
    st.markdown("<h3 style='margin-top: 10px; margin-bottom: 20px;'>Phim trong lịch sử</h3>", unsafe_allow_html=True)

    cols_per_row = 5 
    
    for start_idx in range(0, len(movies), cols_per_row):
        columns = st.columns(cols_per_row)
        
        for col_idx, movie in enumerate(movies[start_idx : start_idx + cols_per_row]):
            with columns[col_idx]:
                render_movie_card(movie, f"history_{start_idx}_{col_idx}")


def render_detail_page(movie_id: int | None) -> None:
    st.button("Về trang chủ", on_click=navigate, args=("home",))
    if movie_id is None:
        st.warning("Không có phim được chọn.")
        return
    movie = api_get(f"/movies/{movie_id}")
    if not movie:
        st.warning("Không tìm thấy phim.")
        return

    title = clean_title(movie.get("title", ""))
    poster_url = movie.get("poster_url") or ""
    genres = str(movie.get("tmdb_genres") or movie.get("genres") or "").replace("|", ", ")
    score = float(movie.get("vote_average") or movie.get("score") or 0.0)
    vote_count = int(movie.get("vote_count") or 0)
    overview = movie.get("overview") or "Chưa có thông tin tóm tắt cho bộ phim này."
    director = movie.get("director") or "Đang cập nhật"
    cast = str(movie.get("cast") or "Đang cập nhật").replace("|", ", ")
    
    explanation_tags = st.session_state.get("current_tags", [])

    if score > 5.0: score = score / 2.0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.5], gap="large")
    
    with c1:
        if poster_url:
            st.image(poster_url, use_container_width=True)
            
    with c2:
        st.markdown(f"<h1 style='font-size: 3rem; margin-bottom: 0;'>{title}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #F5C518; font-size: 1.3rem; font-weight: bold;'>⭐{score:.1f} <span style='color: #aaa; font-weight: normal; font-size: 1rem;'>({vote_count} đánh giá)</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Đạo diễn:</b> {director}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Diễn viên:</b> {cast}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd; margin-bottom: 5px;'><b>Thể loại:</b> {genres}</p>", unsafe_allow_html=True)
        
        if explanation_tags:
            tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; margin-top: 5px;'>"
            for tag in explanation_tags:
                tags_html += f"<span style='background-color: rgba(245, 197, 24, 0.15); color: #F5C518; border: 1px solid rgba(245, 197, 24, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.9rem; font-weight: 500;'>{safe_text(tag)}</span>"
            tags_html += "</div>"
            st.markdown(tags_html, unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: #aaa; line-height: 1.6; margin-top: 15px;'>{overview}</p>", unsafe_allow_html=True)
        
        # ===============================
        # KHOANG ĐÁNH GIÁ PHIM (RATING)
        # ===============================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 10px;'>Đánh giá của bạn</h4>", unsafe_allow_html=True)
        
        user_id = st.session_state.current_user
        
        if user_id is None:
            st.info("Vui lòng chọn User trên thanh Menu để đánh giá phim này.")
        else:
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
                        user_rating = st.selectbox(
                            "Chấm điểm", 
                            options=rating_options,
                            format_func=lambda x: f"{x} ⭐",
                            label_visibility="collapsed"
                        )
                
                with col_btn:
                    if existing_rating is not None:
                        submit_btn = st.form_submit_button("Đã đánh giá", type="secondary", disabled=True, use_container_width=True)
                    else:
                        submit_btn = st.form_submit_button("Gửi đánh giá", type="primary", use_container_width=True)
                    
                if submit_btn and existing_rating is None:
                    payload = {
                        "user_id": user_id,
                        "movie_id": int(float(movie_id)),
                        "rating": float(user_rating)
                    }
                    response = api_post("/rate", payload)
                    
                    if response and response.get("status") == "success":
                        st.toast(f"Đã ghi nhận đánh giá {user_rating} ⭐ thành công!")
                        st.rerun()
                    else:
                        st.toast("Error: Có lỗi xảy ra, chưa thể gửi đánh giá.")
        # ====================================

    st.markdown("<hr style='border-color: #333; margin: 3rem 0 1rem 0;'>", unsafe_allow_html=True)

    similar = movie_list(api_get(f"/movies/{movie_id}/similar", top_k=15))
    render_movie_row("Có thể bạn cũng thích", similar, f"similar_{movie_id}")


def render_metrics_page() -> None:
    st.title("Đánh giá mô hình")
    health = api_get("/health") or {}
    st.json(health)
    metrics = ((api_get("/model-info") or {}).get("model_info") or {}).get("metrics", {})
    rows = []
    for group, values in metrics.items():
        if isinstance(values, dict):
            rows.extend({"nhóm": group, "metric": key, "giá trị": value} for key, value in values.items())
        else:
            rows.append({"nhóm": "baseline", "metric": group, "giá trị": values})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_chatbot_page() -> None:
    st.title("Chatbot tư vấn phim")
    st.caption("Chatbot truy xuất phim từ artifact hiện tại; nếu chưa cấu hình OpenAI key, API sẽ trả lời bằng chế độ local.")

    st.session_state.setdefault("chat_messages", [])
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Bạn muốn xem phim kiểu gì?")
    if not user_message:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm phim phù hợp..."):
            data = api_post("/chat", {"message": user_message, "top_k": 6})
        if not data:
            st.error("Không gọi được chatbot API.")
            return
        answer = str(data.get("answer") or "")
        st.markdown(answer)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        sources = data.get("sources") or []
        if sources:
            st.caption(f"Nguồn gợi ý ({data.get('retrieval_mode') or 'retrieval'}):")
            columns = st.columns(min(3, len(sources)))
            for idx, movie in enumerate(sources[:6]):
                with columns[idx % len(columns)]:
                    poster_url = movie.get("poster_url")
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{clean_title(movie.get('title', ''))}**")
                    st.caption(f"Điểm liên quan: {float(movie.get('score') or 0.0):.3f}")


def main() -> None:
    st.set_page_config(page_title="movierec", layout="wide")
    init_state()
    render_global_styles()
    users = load_users()
    
    page_tabs = st.tabs(["Trang chính", "Chatbot", "Đánh giá", "Trạng thái"])
    
    with page_tabs[0]:
        render_navbar(users)
        
        if st.session_state.page == "search":
            render_search_page()
        elif st.session_state.page == "history":
            render_history_page()
        elif st.session_state.page == "detail":
            render_detail_page(st.session_state.movie_id)
        else:
            render_home_page()
            
    with page_tabs[1]:
        render_chatbot_page()
        
    with page_tabs[2]:
        render_metrics_page()
        
    with page_tabs[3]:
        st.json(api_get("/health") or {})

if __name__ == "__main__":
    main()
