"""Streamlit frontend for the movie recommender."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str, **params: Any) -> Any | None:
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.toast(f"Không gọi được API: {exc}")
        return None


def api_post(path: str, payload: dict[str, Any]) -> Any | None:
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
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
    return str(title or "Không rõ tên").strip()


def navigate(page: str, movie_id: int | None = None) -> None:
    st.session_state.page = page
    if movie_id is not None:
        st.session_state.movie_id = int(movie_id)


def init_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("movie_id", None)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("search_query", "")


def render_navbar(users: list[int]) -> None:
    col_brand, col_search, col_user, col_history = st.columns([1.1, 3.5, 1.4, 1.1])
    with col_brand:
        if st.button("movierec", use_container_width=True):
            navigate("home")
    with col_search:
        query = st.text_input("Tìm phim", value=st.session_state.search_query, label_visibility="collapsed", placeholder="Tìm phim")
        if query != st.session_state.search_query:
            st.session_state.search_query = query
            if query.strip():
                navigate("search")
    with col_user:
        options = ["Guest"] + [str(user) for user in users[:1000]]
        current = "Guest" if st.session_state.current_user is None else str(st.session_state.current_user)
        selected = st.selectbox("User", options, index=options.index(current) if current in options else 0, label_visibility="collapsed")
        st.session_state.current_user = None if selected == "Guest" else int(selected)
    with col_history:
        st.button("Lịch sử", use_container_width=True, disabled=st.session_state.current_user is None, on_click=navigate, args=("history",))


def render_movie_card(movie: dict[str, Any], key: str) -> None:
    movie_id = int(movie.get("movie_id") or movie.get("movieId") or 0)
    title = clean_title(movie.get("title", ""))
    with st.container(border=True):
        poster_url = movie.get("poster_url")
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown("<div style='height:260px;background:#202124;border-radius:6px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        genres = str(movie.get("tmdb_genres") or movie.get("genres") or "").replace("|", ", ")
        if genres:
            st.caption(genres[:90])
        vote = movie.get("vote_average")
        vote_count = movie.get("vote_count")
        if vote is not None:
            st.caption(f"Điểm phim: {float(vote):.1f} ({int(vote_count or 0)} lượt)")
        if movie.get("match_score") is not None:
            st.caption(f"Độ phù hợp: {float(movie['match_score']) * 100:.0f}%")
        if movie.get("user_rating") is not None:
            st.caption(f"Bạn đã chấm: {float(movie['user_rating']):.1f}")
        tags = movie.get("explanation_tags") or []
        if tags:
            st.caption(" | ".join(tags[:2]))
        st.button("Chi tiết", key=f"detail_{key}_{movie_id}", use_container_width=True, disabled=movie_id <= 0, on_click=navigate, args=("detail", movie_id))


def render_movie_row(title: str, movies: list[dict[str, Any]], row_key: str) -> None:
    if not movies:
        return
    st.subheader(title)
    columns = st.columns(min(5, len(movies)))
    for idx, movie in enumerate(movies[:15]):
        with columns[idx % len(columns)]:
            render_movie_card(movie, f"{row_key}_{idx}")


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
    st.title("Gợi ý phim")
    col_title, col_model = st.columns([2, 1])
    with col_model:
        model_label = st.radio("Mô hình", ["Hybrid", "LightGCN", "Content", "Popularity"], horizontal=True)
    user_id = st.session_state.current_user
    if user_id is not None:
        recs = recommend(user_id, model_name=str(model_label).lower(), top_k=15)
        render_movie_row("Được đề xuất cho bạn", recs, "recommended")
    else:
        st.info("Chọn user để xem gợi ý cá nhân hóa.")

    for title, path in [
        ("Phim mới", "/movies/latest"),
        ("Đang thịnh hành", "/movies/trending"),
        ("Đánh giá cao", "/movies/top-rated"),
        ("Hành động", "/movies/genre/Action"),
        ("Hài", "/movies/genre/Comedy"),
    ]:
        render_movie_row(title, movie_list(api_get(path, top_k=15)), path)


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
    rows = []
    for movie in movies:
        rows.append(
            {
                "movie_id": movie.get("movie_id"),
                "title": movie.get("title"),
                "user_rating": movie.get("user_rating"),
                "source": movie.get("history_source"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    render_movie_row("Phim trong lịch sử", movies, "history")


def render_detail_page(movie_id: int | None) -> None:
    st.button("Về trang chủ", on_click=navigate, args=("home",))
    if movie_id is None:
        st.warning("Không có phim được chọn.")
        return
    movie = api_get(f"/movies/{movie_id}")
    if not movie:
        st.warning("Không tìm thấy phim.")
        return

    left, right = st.columns([1, 2])
    with left:
        if movie.get("poster_url"):
            st.image(movie["poster_url"], use_container_width=True)
    with right:
        st.title(clean_title(movie.get("title", "")))
        genres = str(movie.get("tmdb_genres") or movie.get("genres") or "").replace("|", ", ")
        if genres:
            st.caption(genres)
        if movie.get("vote_average") is not None:
            st.caption(f"Điểm phim: {float(movie['vote_average']):.1f} ({int(movie.get('vote_count') or 0)} lượt)")
        if movie.get("director"):
            st.markdown(f"**Đạo diễn:** {movie['director']}")
        if movie.get("cast"):
            st.markdown(f"**Diễn viên:** {str(movie['cast']).replace('|', ', ')}")
        st.write(movie.get("overview") or "Chưa có thông tin tóm tắt.")

        user_id = st.session_state.current_user
        if user_id is None:
            st.info("Chọn user để chấm phim.")
        else:
            existing = api_get(f"/rate/{user_id}/{movie_id}") or {}
            current_rating = existing.get("rating")
            with st.form(f"rating_{user_id}_{movie_id}"):
                rating = st.slider("Đánh giá của bạn", min_value=0.5, max_value=5.0, value=float(current_rating or 4.0), step=0.5)
                submitted = st.form_submit_button("Lưu đánh giá")
            if submitted:
                saved = api_post("/rate", {"user_id": user_id, "movie_id": movie_id, "rating": rating})
                if saved:
                    st.success("Đã lưu đánh giá.")

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
    users = load_users()
    render_navbar(users)
    page_tabs = st.tabs(["Trang chính", "Chatbot", "Đánh giá", "Trạng thái"])
    with page_tabs[0]:
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
