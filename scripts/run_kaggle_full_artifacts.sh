#!/usr/bin/env bash
set -euo pipefail

# Full GPU/Kaggle pipeline for both MovieLens and Letterboxd.
#
# Expected inputs are already present in the repo or mounted Kaggle working dir:
# - data/raw/ml-latest-small
# - data/processed/movie_catalog_enriched.parquet
# - data/processed/letterboxd
# - data/processed/letterboxd/movie_catalog_enriched.parquet
#
# Output artifacts:
# - artifacts/movielens_pdf_clean
# - artifacts/letterboxd_pdf_clean
# - artifacts/movielens_strong
# - artifacts/letterboxd_strong
# - reports/comparison_sbert_pdf_clean_both
# - reports/comparison_sbert_strong_both

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RECOMMENDER_VERBOSE_TRAIN="${RECOMMENDER_VERBOSE_TRAIN:-1}"

LOG_DIR="${LOG_DIR:-logs/kaggle_full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="$LOG_DIR/pipeline.log"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

timestamp() {
  date "+%F %T"
}

log() {
  echo "[$(timestamp)] $*"
}

run_step() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  log "START $name"
  log "LOG $log_file"
  log "CMD $*"
  set +e
  "$@" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    log "FAILED $name status=$status"
    exit "$status"
  fi
  log "DONE $name"
}

log "Project root: $ROOT_DIR"
log "Pipeline log: $PIPELINE_LOG"
log "PYTHONPATH: $PYTHONPATH"
log "RECOMMENDER_VERBOSE_TRAIN: $RECOMMENDER_VERBOSE_TRAIN"

if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  run_step install_requirements "$PYTHON_BIN" -m pip install -q -r requirements.txt
  run_step install_optional_requirements "$PYTHON_BIN" -m pip install -q -r requirements-optional.txt
fi

run_step check_cuda "$PYTHON_BIN" -u -c '
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Enable Kaggle GPU before running full artifacts.")
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0))
'

MOVIELENS_DIR="${MOVIELENS_DIR:-data/raw/ml-latest-small}"
MOVIELENS_CATALOG="${MOVIELENS_CATALOG:-data/processed/movie_catalog_enriched.parquet}"
LETTERBOXD_DIR="${LETTERBOXD_DIR:-data/processed/letterboxd}"
LETTERBOXD_CATALOG="${LETTERBOXD_CATALOG:-data/processed/letterboxd/movie_catalog_enriched.parquet}"

SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/all-mpnet-base-v2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
DIM="${DIM:-128}"
GRID_STEP="${GRID_STEP:-0.05}"
MAX_EASE_ITEMS="${MAX_EASE_ITEMS:-5000}"
MAX_SLIM_ITEMS="${MAX_SLIM_ITEMS:-3000}"
MAX_RANKER_SAMPLES="${MAX_RANKER_SAMPLES:-500000}"

if [[ "${RUN_PDF_CLEAN:-1}" == "1" ]]; then
  run_step train_movielens_pdf_clean "$PYTHON_BIN" -u scripts/train.py \
    --raw-dir "$MOVIELENS_DIR" \
    --enriched-catalog "$MOVIELENS_CATALOG" \
    --artifacts-dir artifacts/movielens_pdf_clean \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --train-lightgcn \
    --train-two-tower \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hybrid-grid-step "$GRID_STEP" \
    --min-rating 4.0

  run_step train_letterboxd_pdf_clean "$PYTHON_BIN" -u scripts/train.py \
    --raw-dir "$LETTERBOXD_DIR" \
    --enriched-catalog "$LETTERBOXD_CATALOG" \
    --artifacts-dir artifacts/letterboxd_pdf_clean \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --train-lightgcn \
    --train-two-tower \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hybrid-grid-step "$GRID_STEP" \
    --min-rating 4.0
fi

if [[ "${RUN_STRONG:-1}" == "1" ]]; then
  run_step train_movielens_strong "$PYTHON_BIN" -u scripts/train_strong_hybrid.py \
    --dataset movielens \
    --raw-dir "$MOVIELENS_DIR" \
    --enriched-catalog "$MOVIELENS_CATALOG" \
    --artifacts-dir artifacts/movielens_strong \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --ranker lightgbm \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --lightgcn-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max-ease-items "$MAX_EASE_ITEMS" \
    --max-ranker-samples "$MAX_RANKER_SAMPLES" \
    --min-rating 4.0

  run_step train_letterboxd_strong "$PYTHON_BIN" -u scripts/train_strong_hybrid.py \
    --dataset letterboxd \
    --raw-dir "$LETTERBOXD_DIR" \
    --enriched-catalog "$LETTERBOXD_CATALOG" \
    --artifacts-dir artifacts/letterboxd_strong \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --ranker lightgbm \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --lightgcn-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max-ease-items "$MAX_EASE_ITEMS" \
    --max-ranker-samples "$MAX_RANKER_SAMPLES" \
    --min-rating 4.0
fi

if [[ "${RUN_COMPARISON_PDF:-1}" == "1" ]]; then
  run_step compare_sbert_pdf_clean_both "$PYTHON_BIN" -u scripts/compare_models.py \
    --dataset both \
    --movielens-dir "$MOVIELENS_DIR" \
    --movielens-enriched-catalog "$MOVIELENS_CATALOG" \
    --letterboxd-dir "$LETTERBOXD_DIR" \
    --letterboxd-enriched-catalog "$LETTERBOXD_CATALOG" \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --preset letterboxd-pdf-clean \
    --models core \
    --k 10 \
    --epochs "$EPOCHS" \
    --mf-dim "$DIM" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hybrid-grid-step "$GRID_STEP" \
    --output-dir reports/comparison_sbert_pdf_clean_both
fi

if [[ "${RUN_COMPARISON_STRONG:-1}" == "1" ]]; then
  run_step compare_sbert_strong_both "$PYTHON_BIN" -u scripts/compare_models.py \
    --dataset both \
    --movielens-dir "$MOVIELENS_DIR" \
    --movielens-enriched-catalog "$MOVIELENS_CATALOG" \
    --letterboxd-dir "$LETTERBOXD_DIR" \
    --letterboxd-enriched-catalog "$LETTERBOXD_CATALOG" \
    --content-backend sbert \
    --sbert-model "$SBERT_MODEL" \
    --preset letterboxd-strong \
    --models full \
    --k 10 \
    --epochs "$EPOCHS" \
    --mf-dim "$DIM" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max-ease-items "$MAX_EASE_ITEMS" \
    --max-slim-items "$MAX_SLIM_ITEMS" \
    --max-ranker-samples "$MAX_RANKER_SAMPLES" \
    --hybrid-grid-step "$GRID_STEP" \
    --output-dir reports/comparison_sbert_strong_both
fi

run_step audit_artifacts "$PYTHON_BIN" -u scripts/audit_artifacts.py

if command -v zip >/dev/null 2>&1; then
  ZIP_PATHS=()
  for path in \
    artifacts/movielens_pdf_clean \
    artifacts/letterboxd_pdf_clean \
    artifacts/movielens_strong \
    artifacts/letterboxd_strong \
    reports/comparison_sbert_pdf_clean_both \
    reports/comparison_sbert_strong_both \
    "$LOG_DIR"
  do
    if [[ -e "$path" ]]; then
      ZIP_PATHS+=("$path")
    else
      log "SKIP zip missing path: $path"
    fi
  done
  if [[ "${#ZIP_PATHS[@]}" -gt 0 ]]; then
    run_step zip_outputs zip -qr artifacts_and_reports_full.zip "${ZIP_PATHS[@]}"
  fi
  log "Created artifacts_and_reports_full.zip"
fi

log "All done"
