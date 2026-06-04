HỆ THỐNG GỢI Ý PHIM HYBRID
==========================

Mục tiêu
--------
Dự án xây dựng hệ thống gợi ý phim kết hợp:
- MovieLens làm dữ liệu tương tác người dùng-phim.
- TMDb làm nguồn làm giàu metadata, poster, overview, đạo diễn và diễn viên.
- LightGCN tối ưu tín hiệu collaborative filtering bằng BPR loss.
- SBERT Two-Tower tạo biểu diễn nội dung để xử lý cold-start.
- FastAPI phục vụ inference, Streamlit làm giao diện demo.


1. Cài đặt local
----------------
Yêu cầu:
- Python 3.10+.
- TMDb API key v3.
- GPU không bắt buộc cho demo nhỏ; Kaggle/GPU được khuyến nghị cho huấn luyện đầy đủ.

Lệnh:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Mở file .env và điền:

    TMDB_API_KEY=...


2. Tải dữ liệu MovieLens
------------------------
Mặc định local dùng ml-latest-small:

    PYTHONPATH=src python scripts/download_movielens.py --dataset ml-latest-small

Dữ liệu sẽ nằm tại:

    data/raw/ml-latest-small

Trên Kaggle có thể dùng ml-20m nếu tài nguyên đủ:

    PYTHONPATH=src python scripts/download_movielens.py --dataset ml-20m

Lưu ý: ml-1m không có links.csv nên không phù hợp với pipeline TMDb bắt buộc.


3. Làm giàu TMDb
----------------
TMDb là bắt buộc trong pipeline chính. Script có cache để chạy lại không gọi API trùng:

    PYTHONPATH=src python scripts/enrich_tmdb.py \
      --raw-dir data/raw/ml-latest-small \
      --output data/processed/movie_catalog_enriched.parquet \
      --cache data/processed/tmdb_cache.json

Smoke test nhanh có thể giới hạn số phim:

    PYTHONPATH=src python scripts/enrich_tmdb.py --limit 200


4. Huấn luyện và export artifacts
---------------------------------
Huấn luyện đầy đủ trên Kaggle/GPU:

    PYTHONPATH=src python scripts/train.py \
      --raw-dir data/raw/ml-latest-small \
      --enriched-catalog data/processed/movie_catalog_enriched.parquet \
      --artifacts-dir artifacts \
      --content-backend sbert \
      --train-lightgcn \
      --epochs 10 \
      --device cuda

Smoke test local nếu chưa cài sentence-transformers/torch:

    PYTHONPATH=src python scripts/train.py \
      --raw-dir data/raw/ml-latest-small \
      --content-backend tfidf \
      --artifacts-dir artifacts

Artifacts xuất ra:
- artifacts/movie_catalog.parquet
- artifacts/user_mapping.json
- artifacts/item_mapping.json
- artifacts/content_embeddings.npy
- artifacts/user_profiles.npy
- artifacts/item_popularity.npy
- artifacts/lightgcn_user_embeddings.npy nếu train LightGCN
- artifacts/lightgcn_item_embeddings.npy nếu train LightGCN
- artifacts/hybrid_config.json
- artifacts/metrics.json

Mặc định script lưu thêm lịch sử item đã xem trong hybrid_config.json để API không gợi ý lại phim đã dùng ở train. Với tập rất lớn có thể tắt bằng:

    --no-store-train-user-items


5. Chạy backend FastAPI
-----------------------

    PYTHONPATH=src:. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
- GET /health
- GET /movies?query=toy&limit=20
- POST /recommendations

Ví dụ request:

    curl -X POST http://localhost:8000/recommendations \
      -H "Content-Type: application/json" \
      -d '{"user_id": 1, "top_k": 10, "session_context": []}'

Ví dụ session cold-start:

    curl -X POST http://localhost:8000/recommendations \
      -H "Content-Type: application/json" \
      -d '{"top_k": 10, "session_context": ["tmdb_862", "ml_1"]}'


6. Chạy giao diện Streamlit
---------------------------
Mở terminal thứ hai:

    API_URL=http://localhost:8000 ARTIFACTS_DIR=artifacts streamlit run app/streamlit_app.py

Truy cập:

    http://localhost:8501


7. Chạy bằng Docker Compose
---------------------------

    cp .env.example .env
    docker compose up --build

Các cổng:
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

Lưu ý: Docker image cài torch và sentence-transformers nên lần build đầu có thể lâu.


8. Chạy tests
-------------

    PYTHONPATH=src:. pytest

Nếu môi trường chưa cài torch hoặc fastapi, các test tương ứng sẽ tự skip.


9. Cấu trúc thư mục
-------------------

    src/recommender/data        Đọc MovieLens, split dữ liệu, TMDb enrichment
    src/recommender/models      SVD, LightGCN, BPR loss, Two-Tower embeddings
    src/recommender/eval        Precision@K, Recall@K, NDCG@K, MRR, RMSE
    src/recommender/inference   Load artifacts và sinh recommendations
    api                         FastAPI backend
    app                         Streamlit frontend
    scripts                     Download, enrich, train/export
    notebooks                   Notebook hướng dẫn Kaggle
    tests                       Unit/integration tests
    reports                     Báo cáo ngắn


10. Ghi chú triển khai Kaggle
-----------------------------
- Lưu TMDB_API_KEY bằng Kaggle Secrets.
- Dùng GPU accelerator nếu train LightGCN và SBERT trên tập lớn.
- Sau khi train, nén thư mục artifacts để tải về máy local:

    zip -r artifacts.zip artifacts

Sau đó giải nén vào repo local và chạy API/UI.
