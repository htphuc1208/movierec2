# Hybrid Movie Recommendation System

This project is a modular hybrid movie recommender with MovieLens-style data, collaborative filtering, content-based metadata encoding, FastAPI, Streamlit, Docker, and offline evaluation.

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

## Train The Funk-SVD Rating Baseline

The independent SVD pipeline trains a biased matrix-factorization model from `models/SVD.py` without changing the API or the default hybrid recommender.

Install PyTorch first. CPU-only install is enough for MovieLens small:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the pipeline:

```bash
python3 scripts/train_svd.py --data-dir data/ml-latest-small --epochs 15 --factors 64 --artifact-path artifacts/svd_ml_latest_small.pt
```

The script reports RMSE/MAE plus Precision@K, Recall@K, NDCG@K, and MRR@K. Artifacts and metrics are written under `artifacts/`, which is ignored by Git.

## Use MovieLens Data

Download MovieLens latest-small when network access is available:

```bash
python3 scripts/download_movielens.py --variant ml-latest-small --output-dir data
```

Then run:

```bash
python3 scripts/train_baseline.py --data-dir data/ml-latest-small
```

For the MovieLens 1M `.dat` format, place `movies.dat`, `ratings.dat`, and `users.dat` in `data/raw`, then run:

```bash
python3 data/run_process.py
```

Processed CSV files are written to `data/processed` and are ignored by Git.

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

The production-style research modules are present in `models/LightGCN.py`, `models/TwoTower.py`, and `models/Loss.py`. They require `requirements-ml.txt`. The default demo uses a numpy Funk-SVD + TF-IDF hybrid implementation so the project can run on a normal laptop without GPU or model downloads.
