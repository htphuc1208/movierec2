# Hybrid Movie Recommendation System

This project implements the plan from `Xay dung He thong Goi y Phim.pdf`: a modular hybrid movie recommender with MovieLens-style data, collaborative filtering, content-based metadata encoding, FastAPI, Streamlit, Docker, and offline evaluation.

The repository is intentionally runnable with a small bundled dataset. When network access and compute are available, replace `data/sample` with MovieLens data and install the optional deep-learning dependencies for LightGCN and SBERT.

## Project Structure

```text
api/                    FastAPI backend
app/                    Streamlit UI
data/                   data loader plus MovieLens-like sample CSV files
evaluation/             Precision, Recall, NDCG, MRR, RMSE
models/                 Hybrid recommender, LightGCN, TwoTower, BPR loss
scripts/                training, download, and TMDb enrichment helpers
tests/                  smoke tests for the local recommender
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu/Debian, install venv support first if `ensurepip is not available`:

```bash
sudo apt install python3.12-venv
python3 -m venv .venv
```

Optional research dependencies for LightGCN and SBERT:

```bash
pip install -r requirements-ml.txt
```

## Run The API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Useful endpoints:

```text
GET  /health
GET  /movies
POST /recommend
```

Example request:

```json
{
  "user_id": 104,
  "top_k": 10,
  "session_context": ["tmdb_862", "1"]
}
```

## Run The UI

```bash
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

The UI tries to call the FastAPI backend at `API_URL` and falls back to local inference if the backend is not running.

## Evaluate The Sample Model

```bash
python3 scripts/train_baseline.py --data-dir data/sample --top-k 10
```

This command trains the lightweight hybrid baseline on the sample CSV files and reports Precision@K, Recall@K, NDCG@K, MRR@K, and RMSE.

## Use MovieLens Data

Download MovieLens latest-small when network access is available:

```bash
python3 scripts/download_movielens.py --variant ml-latest-small --output-dir data
```

Then run:

```bash
python3 scripts/train_baseline.py --data-dir data/ml-latest-small
```

## Optional TMDb Enrichment

Create a TMDb API key and run:

```bash
export TMDB_API_KEY=your_key
python3 scripts/enrich_tmdb.py --data-dir data/ml-latest-small --limit 1000
```

The enrichment script writes `enriched_movies.csv` with poster URL, overview, director, cast, and production metadata when the API is reachable.

## Docker

```bash
docker compose up --build
```

Services:

```text
backend   FastAPI on http://localhost:8000
frontend  Streamlit on http://localhost:8501
postgres  Optional storage service scaffold
```

## Notes

The production-style research modules are present in `models/LightGCN.py`, `models/TwoTower.py`, and `models/Loss.py`. They require `requirements-ml.txt`. The default demo uses a deterministic SVD + TF-IDF hybrid implementation so the project can run on a normal laptop without GPU or model downloads.
