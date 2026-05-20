# 15-Minute Presentation Outline

## 1. Problem And Dataset - Data Engineer - 3 minutes

- Problem: information overload and cold-start in movie discovery.
- Dataset: MovieLens ratings, movies, links, tags.
- Enrichment path: TMDb via `links.csv` to collect overview, poster, director, cast, budget, and revenue.
- Deliverable in repo: `data/dataloader.py`, `data/sample/*.csv`, `scripts/download_movielens.py`, `scripts/enrich_tmdb.py`.

## 2. Graph Collaborative Model - ML Engineer I - 3 minutes

- Baseline: rating matrix and SVD for fast comparison.
- Core research model: LightGCN on the user-item bipartite graph.
- Loss: Bayesian Personalized Ranking with negative sampling.
- Deliverable in repo: `models/LightGCN.py`, `models/Loss.py`.

## 3. Metadata Two-Tower And Hybrid Ranking - ML Engineer II - 3 minutes

- Item tower: title, genres, overview, tagline, director, cast, tags.
- SBERT path: semantic embeddings when `requirements-ml.txt` is installed.
- Fallback path: TF-IDF content vectors for local demo.
- Hybrid score: normalized collaborative score, content score, and popularity score.
- Deliverable in repo: `models/TwoTower.py`, `models/recommender.py`.

## 4. Backend API - System Engineer - 3 minutes

- FastAPI service with `/health`, `/users`, `/movies`, and `/recommend`.
- Request includes `user_id`, `top_k`, `session_context`, and `exclude_seen`.
- Response includes movie IDs, title, score, poster URL, metadata, and reason tags.
- Deliverable in repo: `api/main.py`, `main.py`, `Dockerfile`, `docker-compose.yml`.

## 5. UI And Evaluation - Frontend/QA Engineer - 3 minutes

- Streamlit UI with user selection, session movies, top-K, and poster cards.
- Offline metrics: Precision@K, Recall@K, NDCG@K, MRR@K, RMSE.
- Demo command: `streamlit run app/streamlit_app.py --server.fileWatcherType none`.
- Evaluation command: `python3 scripts/train_baseline.py --data-dir data/sample --top-k 10`.
- Deliverable in repo: `app/streamlit_app.py`, `evaluation/metrics.py`, `tests/test_recommender.py`.
