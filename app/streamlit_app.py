"""Streamlit frontend for the movie recommender."""

from __future__ import annotations

import html
import os
import re
from typing import Any
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

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


def get_session_context() -> list[int]:
    return [int(movie_id) for movie_id in st.session_state.setdefault("session_context", [])]


def add_session_movie(movie: dict[str, Any]) -> None:
    movie_id = parse_movie_id(movie)
    if movie_id <= 0:
        return
    session_ids = st.session_state.setdefault("session_context", [])
    titles = st.session_state.setdefault("session_movie_titles", {})
    posters = st.session_state.setdefault("session_movie_posters", {})
    if movie_id not in session_ids:
        session_ids.append(movie_id)
    titles[str(movie_id)] = clean_title(movie.get("title", f"Movie {movie_id}"))
    posters[str(movie_id)] = movie.get("poster_url") or ""


def remove_session_movie(movie_id: int) -> None:
    session_ids = st.session_state.setdefault("session_context", [])
    st.session_state.session_context = [int(value) for value in session_ids if int(value) != int(movie_id)]
    st.session_state.setdefault("session_movie_titles", {}).pop(str(int(movie_id)), None)
    st.session_state.setdefault("session_movie_posters", {}).pop(str(int(movie_id)), None)


def toggle_session_movie(movie: dict[str, Any]) -> None:
    movie_id = parse_movie_id(movie)
    if movie_id <= 0:
        return
    if movie_id in get_session_context():
        remove_session_movie(movie_id)
    else:
        add_session_movie(movie)


def clear_session_context() -> None:
    st.session_state.session_context = []
    st.session_state.session_movie_titles = {}
    st.session_state.session_movie_posters = {}


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
    st.session_state.setdefault("session_context", [])
    st.session_state.setdefault("session_movie_titles", {})
    st.session_state.setdefault("session_movie_posters", {})
    st.session_state.setdefault("session_weight", 0.65)


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
        if st.button("📽️movierec", use_container_width=True):
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
        
        new_user = None if selected == "Guest" else int(selected)
        
        if new_user != st.session_state.current_user:
            st.session_state.current_user = new_user
            
            st.session_state.chat_messages = []
            st.session_state.bot_is_typing = False
            
            st.rerun()
        
    with col_history:
        st.button("Lịch sử", use_container_width=True, disabled=st.session_state.current_user is None, on_click=navigate, args=("history",))


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        /* Row phim */
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

        /* Tabs Navbar */
        div[data-baseweb="tab-list"] {
            display: flex !important;
            justify-content: center !important;
            gap: 40px !important;
            border-bottom: 2px solid #222 !important; 
            margin-bottom: 2rem !important;
            padding-bottom: 0 !important;
        }
        button[data-baseweb="tab"] {
            font-size: 1.1rem !important; 
            font-weight: 700 !important;
            padding: 1rem 0 !important;
            color: #888 !important; 
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #F5C518 !important; 
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #F5C518 !important;
            height: 3px !important;
            border-radius: 4px 4px 0 0 !important;
        }
        .block-container { padding-top: 2rem !important; }
        
        div[data-testid="stTabs"] > div:first-child {
            position: sticky !important;
            top: 0 !important;
            z-index: 9999 !important;
            padding-top: 20px !important;
            padding-bottom: 5px !important;
            box-shadow: 0px 10px 10px -10px rgba(0,0,0,0.5);
        }

        /* Chatbot */
        .chatbot-marker { display: none !important; }
        div.element-container:has(.chatbot-marker) + div {
            position: fixed !important;
            bottom: 25px !important;
            right: 30px !important;
            z-index: 999999 !important;
            width: auto !important;
        }
        div.element-container:has(.chatbot-marker) + div button {
            background-color: #F5C518 !important;
            color: #000 !important;
            font-weight: bold !important;
            border-radius: 50px !important;
            padding: 10px 22px !important;
            box-shadow: 0 6px 20px rgba(0,0,0,0.5) !important;
            border: none !important;
            transition: transform 0.2s;
        }
        div.element-container:has(.chatbot-marker) + div button:hover { transform: scale(1.05); }
        div[data-testid="stPopoverBody"] {
            background-color: #141414 !important;
            border: 1px solid #333 !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 40px rgba(0,0,0,0.8) !important;
            width: 360px !important;
            padding: 15px !important;
        }

        /* Hero Carousel */
        div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker) {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            overflow-y: hidden !important; 
            scroll-snap-type: x mandatory;
            padding-bottom: 30px !important;
            scroll-behavior: smooth;
        }
        div[data-testid="stHorizontalBlock"]:has(.hero-carousel-marker) > div {
            min-width: 100% !important;
            flex: 0 0 100% !important;
            scroll-snap-align: start;
        }
        div.stButton > button { position: relative !important; z-index: 10 !important; }
        @keyframes blink { 0% {opacity: 0.3;} 50% {opacity: 1;} 100% {opacity: 0.3;} }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero_banner() -> None:
    data = api_get_cached("/movies/trending", top_k=5)
    movies = movie_list(data)
    if not movies:
        return
        
    cols = st.columns(len(movies))
    
    for idx, col in enumerate(cols):
        with col:
            st.markdown('<span class="hero-carousel-marker" style="display:none;"></span>', unsafe_allow_html=True)
            
            m = movies[idx]
            movie_id = parse_movie_id(m)
            
            title = safe_text(clean_title(m.get("title", "Untitled")))
            genres = safe_text(str(m.get("tmdb_genres") or m.get("genres") or "Hấp dẫn").replace("|", ", "))
            
            tmdb_score = float(m.get("vote_average") or 0.0)
                        
            poster = m.get("poster_url") or ""
            if not poster or str(poster).lower() == "nan":
                poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"
                
            html_bg = f"""
            <div style="width: 100%; min-height: 480px; border-radius: 12px; border: 1px solid #222; box-shadow: 0 8px 20px rgba(0,0,0,0.8); display: flex; flex-direction: column; justify-content: flex-end; padding: 45px; padding-bottom: 100px; box-sizing: border-box; background: linear-gradient(to right, rgba(15,15,15,1) 0%, rgba(15,15,15,0.7) 45%, rgba(15,15,15,0) 100%), linear-gradient(to top, rgba(15,15,15,1) 0%, rgba(15,15,15,0.5) 30%, rgba(15,15,15,0) 100%), url('{poster}') no-repeat center top / cover; background-color: #0f0f0f; margin-bottom: -80px; position: relative; z-index: 0;">
            <div style="max-width: 65%;">
            <h1 style="font-size: 3.6rem; margin: 0 0 10px 0; color: white; text-shadow: 2px 2px 5px rgba(0,0,0,1); line-height: 1.15; font-weight: 800; letter-spacing: -1px;">{title}</h1>
            <div style="font-size: 1.15rem; color: #ddd; margin-bottom: 0px; display: flex; align-items: center; gap: 10px; text-shadow: 1px 1px 3px rgba(0,0,0,1);">
            <span style="color: #F5C518; font-weight: bold;">⭐ {tmdb_score:.1f}/10</span>
            <span style="color: #777;">|</span>
            <span style="font-weight: 500;">{genres}</span>
            </div>
            </div>
            </div>
            """
            st.markdown(html_bg, unsafe_allow_html=True)
            
            btn_space, btn_xem, padding_right = st.columns([0.06, 0.2, 0.74])
            
            with btn_xem:
                if movie_id > 0:
                    st.button("Xem chi tiết", type="primary", key=f"hero_det_{idx}_{movie_id}", on_click=navigate, args=("detail", movie_id), width="stretch")
                else:
                    st.button("Xem chi tiết", type="primary", key=f"hero_err_{idx}", disabled=True, width="stretch")

    components.html(
        """
        <script>
        const parentWin = window.parent;
        const parentDoc = parentWin.document;

        function startAutoScroll() {
            const markers = parentDoc.querySelectorAll('.hero-carousel-marker');
            if (markers.length === 0) return;
            
            const carousel = markers[0].closest('div[data-testid="stHorizontalBlock"]');
            if (!carousel) return;

            if (parentWin.heroInterval) {
                clearInterval(parentWin.heroInterval);
            }

            parentWin.heroInterval = setInterval(function() {
                const maxScroll = carousel.scrollWidth - carousel.clientWidth;
                
                if (carousel.scrollLeft >= maxScroll - 10) {
                    carousel.scrollTo({left: 0, behavior: 'smooth'});
                } else {
                    carousel.scrollBy({left: carousel.clientWidth, behavior: 'smooth'});
                }
            }, 5000); 
        }
        
        setTimeout(startAutoScroll, 500);
        </script>
        """,
        height=0,
        width=0,
    )


def render_movie_card(movie: dict[str, Any], key: str, is_row: bool = False) -> None:
    movie_id = parse_movie_id(movie)
    
    poster_url = movie.get("poster_url") or ""
    title = clean_title(movie.get("title", "Untitled"))

    vote_count = int(movie.get("vote_count") or 0)
    match_score = movie.get("match_score")
    genres = str(movie.get("tmdb_genres") or movie.get("genres") or "").replace("|", ", ")
    
    explanation_tags = movie.get("explanation_tags") or []

    tmdb_score = float(movie.get("vote_average") or 0.0)
        
    rating_str = f"⭐ {tmdb_score:.1f}/10" if tmdb_score > 0 else "Chưa có điểm"

    match_html = ""
    if match_score is not None and 0.0 < match_score <= 1.0:
        match_html = f'<span style="color: #46d369; font-weight: bold; font-size: 11px;">{int(match_score * 100)}% Match</span>'

    tags_html = ""
    if explanation_tags:
        for tag in explanation_tags[:2]:
            tags_html += f'<span style="background-color: rgba(245, 197, 24, 0.15); color: #F5C518; border: 1px solid rgba(245, 197, 24, 0.3); border-radius: 4px; padding: 2px 6px; font-size: 10px; display: inline-block; max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{safe_text(tag)}</span>'

    card_height = "135px" if explanation_tags else "75px"

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
<div style="height: {card_height}; margin-bottom: 10px;">
<div style="font-weight: bold; font-size: 15px; line-height: 1.4; height: 44px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; margin-bottom: 4px;">
{title}
</div>
<div style="font-size: 13px; color: #aaa;">
<div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-bottom: 8px;">
{rating_str} • {genres}
</div>
<div style="display: flex; flex-wrap: wrap; gap: 6px; align-items: center;">
{match_html}{tags_html}
</div>
</div>
</div>
"""
        st.markdown(html_content, unsafe_allow_html=True)
        
        if movie_id > 0:
            detail_col, session_col = st.columns([1.2, 0.9])
            with detail_col:
                st.button("Chi tiết", key=f"btn_{key}_{movie_id}", on_click=navigate, args=("detail", movie_id, explanation_tags), use_container_width=True)
            with session_col:
                in_session = movie_id in get_session_context()
                st.button(
                    "Bỏ gu" if in_session else "+ Gu",
                    key=f"taste_{key}_{movie_id}",
                    on_click=toggle_session_movie,
                    args=(movie,),
                    use_container_width=True,
                )
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


def render_session_panel() -> None:
    session_ids = get_session_context()
    if not session_ids:
        st.caption("Mẹo: bấm + Gu trên vài phim để nhận gợi ý theo phiên hiện tại, kể cả khi đang ở Guest.")
        return

    titles = st.session_state.setdefault("session_movie_titles", {})
    posters = st.session_state.setdefault("session_movie_posters", {}) 
    
    with st.container(border=True):
        title_col, slider_col, clear_col = st.columns([3, 2.5, 1], vertical_alignment="center")
        
        with title_col:
            st.markdown(f"<h4 style='margin-bottom: 0px;'>Gu phiên hiện tại <span style='color: #F5C518; font-size: 1.1rem;'>({len(session_ids)} phim)</span></h4>", unsafe_allow_html=True)
            st.caption("Gợi ý sẽ được tinh chỉnh theo các phim này.")
            
        with slider_col:
            st.session_state.session_weight = st.slider(
                "Mức ưu tiên",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("session_weight", 0.65)),
                step=0.05,
                label_visibility="collapsed"
            )
            st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #aaa; margin-top: -12px;'>Mức ưu tiên gu phiên</div>", unsafe_allow_html=True)

        with clear_col:
            st.button("Xóa tất cả", on_click=clear_session_context, use_container_width=True)

        st.markdown("<hr style='margin: 5px 0 15px 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        
        cols_per_row = 3
        
        for start_idx in range(0, len(session_ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, movie_id in enumerate(session_ids[start_idx : start_idx + cols_per_row]):
                title = titles.get(str(movie_id), f'Movie {movie_id}')
                poster = posters.get(str(movie_id), "")
                
                short_title = (title[:30] + '...') if len(title) > 30 else title
                
                if not poster or str(poster).lower() == "nan":
                    poster = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?ixlib=rb-4.0.3&auto=format&fit=crop&w=150&q=80"
                
                with cols[col_idx]:
                    with st.container(border=True):
                        c_info, c_btn = st.columns([0.85, 0.15], vertical_alignment="center")
                        
                        with c_info:
                            st.markdown(
                                f"""
                                <div style='display: flex; align-items: center; gap: 12px; margin-top: -7px; margin-bottom: 0px;'>
                                    <img src="{poster}" style="width: 32px; height: 48px; object-fit: cover; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);">
                                    <div style='font-size: 0.95rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='{safe_text(title)}'>
                                        {safe_text(short_title)}
                                    </div>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        with c_btn:
                            st.button("✖", key=f"session_remove_{movie_id}", on_click=remove_session_movie, args=(movie_id,), use_container_width=True, help="Xóa phim này")


def recommend(
    user_id: int | None,
    model_name: str,
    top_k: int = 15,
    session_context: list[int] | None = None,
    session_weight: float | None = None,
) -> list[dict[str, Any]]:
    if session_context is None:
        session_context = get_session_context()
    if session_weight is None:
        session_weight = float(st.session_state.get("session_weight", 0.65))
    data = api_post(
        "/recommend",
        {
            "user_id": user_id,
            "top_k": top_k,
            "session_context": session_context,
            "session_weight": session_weight,
            "exclude_seen": True,
            "model_name": model_name,
        },
    )
    return movie_list(data)


def render_home_page() -> None:
    render_hero_banner()
    render_session_panel()
    
    user_id = st.session_state.current_user
    session_context = get_session_context()
    session_weight = float(st.session_state.get("session_weight", 0.65))
    
    if user_id is not None:
        col_title, col_model = st.columns([5, 1], vertical_alignment="bottom")
        
        with col_title:
            title = "Được đề xuất cho bạn"
            if session_context and session_weight > 0.0:
                title += " · điều chỉnh theo gu phiên"
            st.markdown(f"<h3 style='margin-top: 15px; margin-bottom: 0;'>{title}</h3>", unsafe_allow_html=True)
            
        with col_model:
            model_label = st.selectbox(
                "Chọn mô hình", 
                ["Hybrid", "LightGCN", "Two-Tower", "Content", "Popularity"],
                format_func=lambda x: f"Mô hình: {x}", 
                help="Chọn thuật toán AI để hệ thống tính toán và đưa ra danh sách phim phù hợp nhất với bạn.",
                label_visibility="collapsed"
            )
        
        recs = recommend(user_id, model_name=str(model_label).lower(), top_k=15, session_context=session_context)
        render_movie_row("", recs, "recommended")
        
    elif session_context and session_weight > 0.0:
        st.markdown("<h3 style='margin-top: 15px; margin-bottom: 0;'>Gợi ý theo gu phiên hiện tại</h3>", unsafe_allow_html=True)
        recs = recommend(None, model_name="hybrid", top_k=15, session_context=session_context)
        render_movie_row("", recs, "session_recommended")

    else:
        st.info("Chọn user hoặc thêm vài phim vào gu phiên hiện tại để xem gợi ý cá nhân hóa.")

    for title, path in [
        ("Phim mới", "/movies/latest"),
        ("Đang thịnh hành", "/movies/trending"),
        ("Đánh giá cao", "/movies/top-rated"),
        ("Hành động", "/movies/genre/Action"),
        ("Hài hước", "/movies/genre/Comedy"),
    ]:
        if title == "Phim mới":
            raw_movies = movie_list(api_get_cached(path, top_k=50))
            today_str = datetime.today().strftime("%Y-%m-%d")
            movies = [
                m for m in raw_movies 
                if not m.get("release_date") or str(m.get("release_date")) <= today_str
            ][:15]
        else:
            movies = movie_list(api_get_cached(path, top_k=15))
            
        render_movie_row(title, movies, path)


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
    vote_count = int(movie.get("vote_count") or 0)
    overview = movie.get("overview") or "Chưa có thông tin tóm tắt cho bộ phim này."
    director = movie.get("director") or "Đang cập nhật"
    cast = str(movie.get("cast") or "Đang cập nhật").replace("|", ", ")
    
    explanation_tags = st.session_state.get("current_tags", [])

    tmdb_score = float(movie.get("vote_average") or 0.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2.5], gap="large")
    
    with c1:
        if poster_url:
            st.image(poster_url, use_container_width=True)
            
    with c2:
        st.markdown(f"<h1 style='font-size: 3rem; margin-bottom: 0;'>{title}</h1>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; align-items: center; margin-top: 10px; margin-bottom: 15px; background-color: rgba(255,255,255,0.05); padding: 10px 15px; border-radius: 8px; width: fit-content;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 0.85rem; color: #aaa; margin-bottom: 2px;">Rating</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #F5C518;">
                        TMDB: {tmdb_score:.1f} <span style="font-size: 0.9rem; color: #777; font-weight: normal;">/10</span>
                    </div>
                </div>
                <div style="width: 1px; height: 35px; background-color: rgba(255,255,255,0.2);"></div>
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
                    <div style="font-size: 0.9rem; color: #888;">{vote_count:,} votes</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        in_session = int(movie_id) in get_session_context()
        st.button(
            "Bỏ khỏi gu phiên hiện tại" if in_session else "Thêm vào gu phiên hiện tại",
            key=f"detail_session_{movie_id}",
            on_click=toggle_session_movie,
            args=(movie,),
        )
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd; margin-top: 10px;'><b>Đạo diễn:</b> {director}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd;'><b>Diễn viên:</b> {cast}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 1.1rem; color: #ddd; margin-bottom: 5px;'><b>Thể loại:</b> {genres}</p>", unsafe_allow_html=True)
        
        if explanation_tags:
            tags_html = "<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; margin-top: 5px;'>"
            for tag in explanation_tags:
                tags_html += f"<span style='background-color: rgba(245, 197, 24, 0.15); color: #F5C518; border: 1px solid rgba(245, 197, 24, 0.3); border-radius: 6px; padding: 4px 10px; font-size: 0.9rem; font-weight: 500;'>{safe_text(tag)}</span>"
            tags_html += "</div>"
            st.markdown(tags_html, unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: #aaa; line-height: 1.6; margin-top: 15px;'>{overview}</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom: 10px;'>Đánh giá của bạn</h4>", unsafe_allow_html=True)
        
        user_id = st.session_state.current_user
        
        if user_id is None:
            st.info("Vui lòng chọn User trên thanh Menu để đánh giá phim này.")
        else:
            existing_rating_data = api_get(f"/rate/{user_id}/{movie_id}")
            existing_rating = existing_rating_data.get("rating") if existing_rating_data else None
            
            rating_options = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5,
                              5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
            
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

    st.markdown("<hr style='border-color: #333; margin: 3rem 0 1rem 0;'>", unsafe_allow_html=True)

    similar = movie_list(api_get(f"/movies/{movie_id}/similar", top_k=15))
    render_movie_row("Có thể bạn cũng thích", similar, f"similar_{movie_id}")


def render_metrics_page() -> None:
    st.title("Đánh giá mô hình")
    metrics = ((api_get("/model-info") or {}).get("model_info") or {}).get("metrics", {})
    rows = []
    for group, values in metrics.items():
        if isinstance(values, dict):
            rows.extend({"nhóm": group, "metric": key, "giá trị": value} for key, value in values.items())
        else:
            rows.append({"nhóm": "baseline", "metric": group, "giá trị": values})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def handle_chat_submit():
    user_message = st.session_state.chat_input_widget
    if user_message.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_message})
        st.session_state.chat_input_widget = "" 
        st.session_state.bot_is_typing = True
        st.session_state.pending_message = user_message


def render_floating_chatbot() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("bot_is_typing", False)

    st.markdown('<div class="chatbot-marker"></div>', unsafe_allow_html=True)
    
    with st.popover("💬 Trợ lý AI"):
        st.markdown("<div style='font-weight: bold; font-size: 1.1rem; color: #F5C518; margin-bottom: 12px; border-bottom: 1px solid #333; padding-bottom: 8px;'>🤖 AI Tư vấn Phim</div>", unsafe_allow_html=True)
        
        chat_html = "<div style='height: 280px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding-right: 5px; margin-bottom: 12px;'>"
        if not st.session_state.chat_messages:
            chat_html += "<div style='color: #777; font-size: 0.9rem; text-align: center; margin-top: 100px;'>Chào bác! Bác muốn tìm xem bộ phim kiểu như thế nào?</div>"
        else:
            for msg in st.session_state.chat_messages:
                if msg["role"] == "user":
                    chat_html += f"<div style='align-self: flex-end; background-color: #0084ff; color: white; padding: 6px 12px; border-radius: 14px; max-width: 80%; font-size: 0.9rem; word-break: break-word;'>{safe_text(msg['content'])}</div>"
                else:
                    chat_html += f"<div style='align-self: flex-start; background-color: #262626; color: #eee; padding: 6px 12px; border-radius: 14px; max-width: 80%; font-size: 0.9rem; word-break: break-word;'>{safe_text(msg['content'])}</div>"
        
        if st.session_state.bot_is_typing:
            chat_html += "<div style='align-self: flex-start; background-color: transparent; color: #F5C518; padding: 0px 5px; font-size: 0.85rem; font-style: italic; animation: blink 1.5s infinite;'>AI đang suy nghĩ...</div>"
            
        chat_html += "</div>"
        
        st.markdown(chat_html, unsafe_allow_html=True)
        
        st.text_input(
            "Tin nhắn",
            key="chat_input_widget",
            placeholder="Đợi AI trả lời nhé..." if st.session_state.bot_is_typing else "Hỏi AI phim gì đó...",
            label_visibility="collapsed",
            on_change=handle_chat_submit,
            disabled=st.session_state.bot_is_typing
        )

        if st.session_state.bot_is_typing:
            data = api_post("/chat", {"message": st.session_state.pending_message, "top_k": 3})
            
            if data:
                answer = str(data.get("answer") or "")
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            else:
                st.session_state.chat_messages.append({"role": "assistant", "content": "Xin lỗi bác, tôi đang bị nghẽn mạng chút. Bác thử lại sau nhé!"})
            
            st.session_state.bot_is_typing = False
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="movierec", layout="wide")
    init_state()
    render_global_styles()
    users = load_users()
    
    page_tabs = st.tabs(["Trang chính", "Đánh giá", "Trạng thái"])
    
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
        render_metrics_page()
        
    with page_tabs[2]:
        st.json(api_get("/health") or {})

    render_floating_chatbot()

if __name__ == "__main__":
    main()