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
      --train-two-tower \
      --epochs 10 \
      --hybrid-grid-step 0.05 \
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
- artifacts/two_tower_user_embeddings.npy nếu train learned Two-Tower
- artifacts/two_tower_item_embeddings.npy nếu train learned Two-Tower
- artifacts/hybrid_config.json
- artifacts/metrics.json

Mặc định script lưu thêm lịch sử item đã xem trong hybrid_config.json để API không gợi ý lại phim đã dùng ở train. Với tập rất lớn có thể tắt bằng:

    --no-store-train-user-items


5. So sánh mô hình
------------------
Runner comparison không thay đổi artifact/API hiện tại. Nó chạy nhiều mô hình trên cùng split train/val/test và xuất báo cáo ranking metrics.

Chạy core suite trên cả MovieLens và Letterboxd processed:

    python3 scripts/compare_models.py \
      --dataset both \
      --movielens-dir data/raw/ml-latest-small \
      --letterboxd-dir data/processed/letterboxd \
      --content-backend tfidf \
      --models core

Output mặc định:

    reports/comparison/comparison_results.csv
    reports/comparison/comparison_results.json
    reports/comparison/comparison_summary.md

Core suite gồm random, popularity, ItemKNN, UserKNN, SVD ranking, EASE, TF-IDF/SBERT content-only, BPR-MF, LightGCN-only, learned Two-Tower, weighted hybrid và learned SGD hybrid ranker. Nếu interpreter thiếu torch thì các model Torch sẽ được ghi skipped_dependency, không làm fail toàn bộ runner.

Chạy full suite nếu đã cài optional native packages:

    pip install -r requirements-optional.txt
    python3 scripts/compare_models.py --dataset both --models full --content-backend tfidf

Full suite bật thêm SLIM ElasticNet, implicit ALS, LightFM WARP và NeuMF. Với dataset lớn, giới hạn SLIM/EASE để tránh chạy quá lâu:

    python3 scripts/compare_models.py --models full --max-slim-items 1000 --max-ease-items 8000

Dùng SBERT khi có GPU hoặc chạy Kaggle:

    python3 scripts/compare_models.py --dataset both --content-backend sbert --device cuda

Hai preset chính cho Letterboxd:

    PYTHONPATH=src:. python scripts/compare_models.py \
      --dataset letterboxd \
      --letterboxd-dir data/processed/letterboxd \
      --letterboxd-enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
      --content-backend sbert \
      --sbert-model sentence-transformers/all-mpnet-base-v2 \
      --preset letterboxd-pdf-clean \
      --k 10 \
      --device cuda \
      --output-dir reports/comparison_letterboxd_pdf_clean

    PYTHONPATH=src:. python scripts/compare_models.py \
      --dataset letterboxd \
      --letterboxd-dir data/processed/letterboxd \
      --letterboxd-enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
      --content-backend sbert \
      --sbert-model sentence-transformers/all-mpnet-base-v2 \
      --preset letterboxd-strong \
      --models full \
      --k 10 \
      --epochs 100 \
      --mf-dim 128 \
      --batch-size 8192 \
      --device cuda \
      --max-ease-items 5000 \
      --max-slim-items 3000 \
      --max-ranker-samples 500000 \
      --output-dir reports/comparison_letterboxd_strong

Preset letterboxd-pdf-clean giữ phương pháp sạch: LightGCN, learned Two-Tower từ SBERT/TF-IDF metadata, content-only và popularity, tune weighted sum bằng validation. Preset letterboxd-strong thêm candidate generators và learned ranker để tối ưu NDCG@10; báo cáo vẫn giữ từng baseline riêng để so sánh.

Markdown report hiển thị metric 4 chữ số và có thêm slice metrics cho sparse users, warm users, long-tail items và head items.


6. Chạy backend FastAPI
-----------------------

    PYTHONPATH=src:. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
- GET /health
- GET /movies?query=toy&limit=20
- POST /recommendations
- POST /recommend là alias tương thích UI cũ
- GET /users
- GET /users/{user_id}/history
- GET /movies/trending
- GET /movies/top-rated
- GET /movies/latest
- GET /movies/genre/{genre}
- GET /movies/{movie_id}
- GET /movies/{movie_id}/similar
- GET /model-info
- POST /rate
- GET /rate/{user_id}/{movie_id}

Rating gửi từ UI/API được lưu vào sidecar CSV, mặc định:

    artifacts/runtime/user_ratings.csv

Có thể đổi bằng biến môi trường:

    RATINGS_STORE_PATH=artifacts/runtime/user_ratings.csv

Ví dụ request:

    curl -X POST http://localhost:8000/recommendations \
      -H "Content-Type: application/json" \
      -d '{"user_id": 1, "top_k": 10, "session_context": [], "model_name": "hybrid"}'

Ví dụ session cold-start:

    curl -X POST http://localhost:8000/recommendations \
      -H "Content-Type: application/json" \
      -d '{"top_k": 10, "session_context": ["tmdb_862", "ml_1"]}'


7. Chạy giao diện Streamlit
---------------------------
Mở terminal thứ hai:

    API_URL=http://localhost:8000 ARTIFACTS_DIR=artifacts streamlit run app/streamlit_app.py

Truy cập:

    http://localhost:8501

Dashboard EDA chạy riêng:

    PYTHONPATH=src:. streamlit run app/eda_app.py --server.port 8502

Truy cập:

    http://localhost:8502


8. Chạy bằng Docker Compose
---------------------------

    cp .env.example .env
    docker compose up --build

Các cổng:
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501
- EDA dashboard: http://localhost:8502

Lưu ý: Docker image cài torch và sentence-transformers nên lần build đầu có thể lâu.


9. Chạy tests
-------------

    PYTHONPATH=src:. pytest

Nếu môi trường chưa cài torch hoặc fastapi, các test tương ứng sẽ tự skip.


10. Cấu trúc thư mục
-------------------

    src/recommender/data        Đọc MovieLens, split dữ liệu, TMDb enrichment
    src/recommender/experiments Runner so sánh mô hình offline
    src/recommender/models      SVD, KNN, EASE, MF, LightGCN, Two-Tower, hybrid ranker
    src/recommender/eval        Precision@K, Recall@K, NDCG@K, MRR, RMSE
    src/recommender/inference   Load artifacts và sinh recommendations
    api                         FastAPI backend
    app                         Streamlit frontend
    scripts                     Download, enrich, train/export
    notebooks                   Notebook hướng dẫn Kaggle
    tests                       Unit/integration tests
    reports                     Báo cáo ngắn


11. Ghi chú triển khai Kaggle
-----------------------------
- Lưu TMDB_API_KEY bằng Kaggle Secrets.
- Dùng GPU accelerator nếu train LightGCN và SBERT trên tập lớn.
- Với comparison mạnh, cài optional packages:

    pip install -q -r requirements.txt
    pip install -q -r requirements-optional.txt

- Sau khi train, nén thư mục artifacts để tải về máy local:

    zip -r artifacts.zip artifacts

Sau đó giải nén vào repo local và chạy API/UI.


12. Chạy riêng bước enrich TMDb trên Kaggle
-------------------------------------------
Khi mạng local bị reset/refused với api.themoviedb.org, có thể submit riêng bước enrich lên Kaggle.

Chuẩn bị:
- Tạo Kaggle API token và đặt tại ~/.kaggle/kaggle.json, hoặc export KAGGLE_USERNAME/KAGGLE_KEY.
- Trong Kaggle UI, tạo Secret tên TMDB_API_KEY chứa TMDb API key.

Submit job enrich MovieLens small:

    PYTHONPATH=src:. python scripts/submit_kaggle_enrich.py \
      --username <kaggle_username> \
      --dataset ml-latest-small

Chạy thử ít phim trước:

    PYTHONPATH=src:. python scripts/submit_kaggle_enrich.py \
      --username <kaggle_username> \
      --dataset ml-latest-small \
      --limit 100

Tải output sau khi kernel hoàn tất:

    kaggle kernels output <kaggle_username>/movierec3-tmdb-enrich -p kaggle_outputs

File cần lấy về local:

    kaggle_outputs/data/processed/movie_catalog_enriched.parquet
    kaggle_outputs/data/processed/tmdb_cache.json


13. Tích hợp dữ liệu Letterboxd
------------------------------
Dữ liệu Letterboxd không có timestamp hành vi đủ tin cậy. File created_at là thời điểm crawler ghi dữ liệu, không phải thời điểm người dùng xem/chấm phim, nên pipeline dùng split random ổn định theo từng user thông qua timestamp synthetic.

Chuẩn bị bản CF-ready sang format tương thích MovieLens:

    PYTHONPATH=src:. python scripts/prepare_letterboxd.py \
      --raw-dir data/letterboxd/data/raw \
      --output-dir data/processed/letterboxd \
      --split cf \
      --rating-policy implicit \
      --seed 42

Output:

    data/processed/letterboxd/ratings.csv
    data/processed/letterboxd/movies.csv
    data/processed/letterboxd/links.csv
    data/processed/letterboxd/letterboxd_user_mapping.csv
    data/processed/letterboxd/letterboxd_movie_mapping.csv
    data/processed/letterboxd/letterboxd_interactions_debug.csv
    data/processed/letterboxd/letterboxd_prepare_summary.json
    data/processed/letterboxd/movie_catalog_enriched.parquet

Rating policy:
- implicit: dùng implicit_score. Phù hợp cho LightGCN/BPR ranking. Chạy train với --min-rating 4.0 để lấy liked/favorite/high-rating làm positive.
- explicit: chỉ dùng interaction_type == rating và rating thật. Phù hợp hơn nếu muốn baseline explicit rating, nhưng mất dữ liệu.

Enrich TMDb cho Letterboxd bằng schema chung với MovieLens:

    PYTHONPATH=src:. python scripts/prepare_letterboxd.py \
      --raw-dir data/letterboxd/data/raw \
      --output-dir data/processed/letterboxd \
      --split cf \
      --rating-policy implicit \
      --enrich-tmdb \
      --sleep-seconds 0.5 \
      --timeout 60 \
      --max-retries 8

Nếu mạng local tới api.themoviedb.org bị reset/refused, chạy bước prepare không enrich ở local, rồi chạy enrich trên Kaggle/cloud hoặc mạng khác.

Train từ Letterboxd đã prepare:

    PYTHONPATH=src:. python scripts/train.py \
      --raw-dir data/processed/letterboxd \
      --enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
      --artifacts-dir artifacts/letterboxd \
      --content-backend tfidf \
      --min-rating 4.0

Khi đã có sentence-transformers/torch và môi trường mạnh hơn:

    PYTHONPATH=src:. python scripts/train.py \
      --raw-dir data/processed/letterboxd \
      --enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
      --artifacts-dir artifacts/letterboxd_pdf_clean \
      --content-backend sbert \
      --train-lightgcn \
      --train-two-tower \
      --lightgcn-dim 128 \
      --lightgcn-layers 3 \
      --epochs 100 \
      --batch-size 8192 \
      --device cuda \
      --hybrid-grid-step 0.05 \
      --min-rating 4.0

Export artifact strongest ranker cho Letterboxd:

    PYTHONPATH=src:. python scripts/train_strong_hybrid.py \
      --dataset letterboxd \
      --raw-dir data/processed/letterboxd \
      --enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
      --artifacts-dir artifacts/letterboxd_strong \
      --content-backend sbert \
      --sbert-model sentence-transformers/all-mpnet-base-v2 \
      --ranker lightgbm \
      --lightgcn-dim 128 \
      --lightgcn-layers 3 \
      --lightgcn-epochs 100 \
      --batch-size 8192 \
      --device cuda \
      --min-rating 4.0

Strong artifact thêm:
- ranker.joblib nếu joblib có thể serialize ranker.
- component_score_config.json.
- two_tower_user_embeddings.npy và two_tower_item_embeddings.npy nếu learned Two-Tower train thành công.

API sẽ tự dùng ranker nếu hybrid_config.json có model_type strong_ranker và ranker.joblib load được. Nếu thiếu lightgbm/joblib hoặc file ranker, API fallback về hybrid scores từ LightGCN, Two-Tower, content và popularity.


14. Chatbot, EDA và embedding visualization
-------------------------------------------
Các tính năng này được port từ các branch remote embedding_visualization, chatbot và EDA, nhưng đã chỉnh lại để dùng artifact/schema hiện tại.

Chatbot RAG:
- API endpoint: POST /chat
- Streamlit chính có tab Chatbot.
- Nếu OPENAI_API_KEY có trong .env, chatbot gọi OpenAI model trong CHAT_MODEL.
- Nếu chưa có OPENAI_API_KEY, chatbot vẫn trả lời local bằng các phim retrieved từ catalog.
- Retriever dùng SBERT nếu artifact được train bằng content_backend=sbert; nếu không, fallback sang lexical TF-IDF trên metadata phim.

Cấu hình .env:

    OPENAI_API_KEY=
    CHAT_MODEL=gpt-4.1-mini

Ví dụ gọi API:

    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "Tôi muốn xem phim khoa học viễn tưởng về không gian", "top_k": 6}'

EDA dashboard:

    PYTHONPATH=src:. streamlit run app/eda_app.py

Dashboard này đọc:
- MovieLens ratings: data/raw/ml-latest-small/ratings.csv
- MovieLens catalog: data/processed/movie_catalog_enriched.parquet
- Letterboxd ratings: data/processed/letterboxd/ratings.csv
- Letterboxd catalog: data/processed/letterboxd/movie_catalog_enriched.parquet

Nội dung EDA gồm thống kê tổng quan, phân phối rating, metadata TMDb, top thể loại/top phim và user segmentation bằng KMeans/GMM.

Embedding visualization:

    PYTHONPATH=src:. python scripts/visualize_embeddings.py \
      --artifacts-dir artifacts/letterboxd_pdf_clean \
      --output-dir reports/embedding_visualization_letterboxd \
      --method tsne \
      --sample-size 2500 \
      --top-genres 8

Output:

    reports/embedding_visualization_letterboxd/embedding_visualization.html
    reports/embedding_visualization_letterboxd/embedding_visualization_points.csv
