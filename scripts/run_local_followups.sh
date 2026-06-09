#!/usr/bin/env bash
set -euo pipefail

# Local CPU checks and reports. This does not create the final SBERT/GPU
# artifacts; use scripts/run_kaggle_full_artifacts.sh for that.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
export RATINGS_STORE_PATH="${RATINGS_STORE_PATH:-artifacts/runtime/local_followup_ratings.csv}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

"$PYTHON_BIN" scripts/audit_artifacts.py
"$PYTHON_BIN" -m compileall -q api app scripts src tests
"$PYTHON_BIN" -m pytest -rs

if [[ "${RUN_COMPARISON:-0}" == "1" ]]; then
  "$PYTHON_BIN" scripts/compare_models.py \
    --dataset both \
    --movielens-dir data/raw/ml-latest-small \
    --movielens-enriched-catalog data/processed/movie_catalog_enriched.parquet \
    --letterboxd-dir data/processed/letterboxd \
    --letterboxd-enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
    --content-backend tfidf \
    --models core \
    --epochs "${EPOCHS:-5}" \
    --batch-size "${BATCH_SIZE:-8192}" \
    --device cpu \
    --output-dir reports/comparison_tfidf_core_both
fi

"$PYTHON_BIN" - <<'PY'
from fastapi.testclient import TestClient

from api.main import app, get_chatbot, get_rating_store, get_recommender

get_recommender.cache_clear()
get_rating_store.cache_clear()
get_chatbot.cache_clear()
client = TestClient(app)
users = client.get("/users")
users.raise_for_status()
first_user = users.json()["users"][0]

checks = [
    ("health", client.get("/health")),
    ("model-info", client.get("/model-info")),
    ("movies-search", client.get("/movies", params={"query": "Toy", "limit": 3})),
    ("recommendations", client.post("/recommendations", json={"user_id": first_user, "top_k": 5, "model_name": "hybrid"})),
    ("trending", client.get("/movies/trending", params={"top_k": 5})),
    ("chat", client.post("/chat", json={"message": "Tôi muốn phim khoa học viễn tưởng", "top_k": 3})),
]
for name, response in checks:
    response.raise_for_status()
    print(f"{name}: {response.status_code}")
PY

"$PYTHON_BIN" scripts/audit_artifacts.py
