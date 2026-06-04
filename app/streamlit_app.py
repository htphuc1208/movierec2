"""Streamlit frontend for the movie recommender."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def api_get(path: str, **params: Any) -> Any:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def movie_options(query: str) -> list[dict[str, Any]]:
    if not query:
        return []
    try:
        return api_get("/movies", query=query, limit=20)
    except requests.RequestException:
        return []


def render_recommendations(items: list[dict[str, Any]]) -> None:
    if not items:
        st.warning("Chưa có kết quả.")
        return
    cols = st.columns(5)
    for idx, item in enumerate(items):
        with cols[idx % 5]:
            if item.get("poster_url"):
                st.image(item["poster_url"], use_container_width=True)
            st.markdown(f"**{item.get('title', '')}**")
            if item.get("score") is not None:
                st.caption(f"Điểm: {item['score']:.3f}")
            tags = item.get("explanation_tags") or []
            if tags:
                st.caption(" | ".join(tags))


def main() -> None:
    st.set_page_config(page_title="Gợi ý phim hybrid", layout="wide")
    st.title("Hệ thống gợi ý phim hybrid")

    tab_recs, tab_metrics, tab_status = st.tabs(["Gợi ý", "Đánh giá", "Trạng thái"])

    with tab_recs:
        left, right = st.columns([1, 2])
        with left:
            mode = st.radio("Nguồn gợi ý", ["User ID", "Phiên xem"], horizontal=True)
            top_k = st.slider("Số lượng", min_value=5, max_value=30, value=10, step=5)
            user_id = None
            session_context: list[str] = []

            if mode == "User ID":
                user_id_text = st.text_input("User ID MovieLens", value="1")
                user_id = int(user_id_text) if user_id_text.strip().isdigit() else None
            else:
                query = st.text_input("Tìm phim")
                choices = movie_options(query)
                labels = {f"{movie['title']} (ml_{movie['movie_id']})": f"ml_{movie['movie_id']}" for movie in choices}
                selected = st.multiselect("Phim trong phiên", list(labels.keys()))
                session_context = [labels[label] for label in selected]

            run = st.button("Tạo gợi ý", type="primary")

        with right:
            if run:
                try:
                    payload = {"user_id": user_id, "top_k": top_k, "session_context": session_context}
                    data = api_post("/recommendations", payload)
                    render_recommendations(data.get("recommendations", []))
                except requests.RequestException as exc:
                    st.error(f"Không gọi được API: {exc}")

    with tab_metrics:
        try:
            status = api_get("/health")
            if not status.get("artifacts", {}).get("ready"):
                st.warning("Chưa có artifacts huấn luyện.")
            else:
                import json
                from pathlib import Path

                metrics_path = Path(os.getenv("ARTIFACTS_DIR", "artifacts")) / "metrics.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    rows = []
                    for group, values in metrics.items():
                        if isinstance(values, dict):
                            rows.extend({"nhóm": group, "metric": key, "giá trị": value} for key, value in values.items())
                        else:
                            rows.append({"nhóm": "baseline", "metric": group, "giá trị": values})
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("metrics.json không tồn tại trong artifacts.")
        except requests.RequestException as exc:
            st.error(f"Không gọi được API: {exc}")

    with tab_status:
        try:
            st.json(api_get("/health"))
        except requests.RequestException as exc:
            st.error(f"Không gọi được API: {exc}")


if __name__ == "__main__":
    main()
