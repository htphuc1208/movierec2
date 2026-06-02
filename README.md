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
GET  /model-info
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

To export an API-loadable artifact:

```bash
python3 scripts/train_baseline.py --data-dir data/ml-latest-small --top-k 10 --artifact-dir artifacts/recommender/latest --dataset-name ml-latest-small
MOVIEREC_DATA_DIR=data/ml-latest-small MOVIEREC_ARTIFACT_DIR=artifacts/recommender/latest uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Train The PyTorch SVD Rating Baseline

The default hybrid recommender now uses the same biased PyTorch SVD engine when Torch is installed, then stores only numpy arrays for inference. If Torch is unavailable, it falls back to the older numpy Funk-SVD implementation so the API/UI can still start.

Install PyTorch first. CPU-only install is enough for MovieLens small:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run the pipeline:

```bash
python3 scripts/train_svd.py \
  --data-dir data/ml-latest-small \
  --artifact-path artifacts/svd_ml_latest_small.pt \
  --recommender-artifact-dir artifacts/recommender/svd-ml-latest-small \
  --dataset-name ml-latest-small
```

The script reports RMSE/MAE plus Precision@K, Recall@K, NDCG@K, and MRR@K. Defaults are tuned for `ml-latest-small` (`factors=24`, small initialization, shrinkage bias priors, explicit L2 regularization, validation early stopping). The `.pt` artifact keeps the PyTorch checkpoint for research. The `--recommender-artifact-dir` output writes a lightweight `manifest.json` plus `collaborative.npz` that the API/UI can load without importing Torch:

```bash
MOVIEREC_DATA_DIR=data/ml-latest-small \
MOVIEREC_ARTIFACT_DIR=artifacts/recommender/svd-ml-latest-small \
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Train Or Test The Content Model

Content-based recommendation logic lives in `models/TwoTower.py`. It supports the repository's `movieId`/`userId` schema and falls back to TF-IDF when SBERT is not installed.

```python
from data import MovieLensDataLoader
from models import TFIDFRecommender

bundle = MovieLensDataLoader("data/sample").load()
model = TFIDFRecommender().fit(bundle.movies, bundle.tags)
model.recommend_similar_movies(movie_id=1, top_k=5)
model.recommend_for_user(user_id=104, user_history=bundle.ratings, top_k=5)
```

For SBERT experiments, install `requirements-ml.txt` and use `SBERTRecommender`. `TwoTowerModel` is also available as a small PyTorch projection layer for SBERT/user-profile vectors, but the default content runtime remains artifact-light TF-IDF. Generated vectors and similarity matrices should be saved under `artifacts/`.

## Use MovieLens Data

Download MovieLens latest-small when network access is available:

```bash
python3 scripts/download_movielens.py --variant ml-latest-small --output-dir data
```

Then run:

```bash
python3 scripts/train_baseline.py --data-dir data/ml-latest-small
```

## RecBole Benchmark And Hybrid Tuning

RecBole is isolated in a Python 3.10 trainer image because some pinned RecBole dependencies do not install cleanly in newer local Python versions.

```bash
docker compose --profile train build trainer
docker compose --profile train run --rm trainer python scripts/benchmark_recbole.py --data-dir data/ml-latest-small --models Pop,ItemKNN,BPR,LightGCN --top-k 10 20
docker compose --profile train run --rm trainer python scripts/tune_hybrid.py --data-dir data/ml-latest-small --cf-model LightGCN --content-backend tfidf --output-dir artifacts/recommender/latest
```

Benchmark reports are written to `artifacts/benchmarks/`. The tuned hybrid artifact is written to `artifacts/recommender/latest/` and can be loaded by setting `MOVIEREC_ARTIFACT_DIR`.

Native PyTorch LightGCN training is also available:

```bash
python3 scripts/train_lightgcn.py \
  --data-dir data/ml-latest-small \
  --epochs 50 \
  --artifact-path artifacts/lightgcn_ml_latest_small.pt \
  --recommender-artifact-dir artifacts/recommender/lightgcn-ml-latest-small \
  --dataset-name ml-latest-small
```

The exported recommender artifact stores final LightGCN embeddings in the same `manifest.json` plus `collaborative.npz` format used by the API/UI.

For the MovieLens 1M `.dat` format, place `movies.dat`, `ratings.dat`, and `users.dat` in `data/raw`, then run:

```bash
python3 data/run_process.py
```

Processed CSV files are written to `data/processed` and are ignored by Git.

## Optional TMDb Enrichment

Create a TMDb API key and run:

```bash
export TMDB_API_KEY=your_key
python3 scripts/enrich_tmdb.py --data-dir data/ml-latest-small --retry-empty --sleep 0.3
```

The enrichment script writes `enriched_movies.csv` with poster URL, overview, director, cast, and production metadata when the API is reachable.
For the current data policy, warm/cold split handling, TMDb keywords, Tag Genome, and benchmark dataset choices, see [docs/data_strategy.md](docs/data_strategy.md).

## Letterboxd Crawl Data

The `crawl/` folder contains an experimental Letterboxd crawler and CSV outputs merged from the remote crawl branch. This data is crawler-specific and is not a drop-in MovieLens replacement yet.

```bash
python3 crawl/crawl_letterboxd_movie_centric.py --resume
python3 crawl/enrich_tmdb.py --api-key "$TMDB_API_KEY" --data-dir crawl/data/raw
```

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

The production-style research modules are present in `models/LightGCN.py`, `models/TwoTower.py`, and `models/Loss.py`. They require `requirements-ml.txt`. The default demo uses PyTorch SVD + TF-IDF when Torch is installed, and falls back to numpy Funk-SVD + TF-IDF when running in a minimal API/UI environment without Torch.
