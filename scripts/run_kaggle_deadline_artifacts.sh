#!/usr/bin/env bash
set -euo pipefail

# Deadline-oriented Kaggle pipeline.
#
# Goal: produce API-ready weighted hybrid artifacts fast enough for a
# near-deadline submission. This intentionally skips the expensive full
# comparison and strong-ranker benchmark in run_kaggle_full_artifacts.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RECOMMENDER_VERBOSE_TRAIN="${RECOMMENDER_VERBOSE_TRAIN:-1}"

LOG_DIR="${LOG_DIR:-logs/kaggle_deadline_$(date +%Y%m%d_%H%M%S)}"
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

if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  run_step install_requirements "$PYTHON_BIN" -m pip install -q -r requirements.txt
  run_step uninstall_incompatible_torch_extras "$PYTHON_BIN" -m pip uninstall -y torchvision torchaudio torchcodec
fi

CONTENT_BACKEND="${CONTENT_BACKEND:-sbert}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
DIM="${DIM:-64}"
GRID_STEP="${GRID_STEP:-0.25}"
MIN_RATING="${MIN_RATING:-4.0}"

RUN_MOVIELENS="${RUN_MOVIELENS:-1}"
RUN_LETTERBOXD="${RUN_LETTERBOXD:-1}"

MOVIELENS_DIR="${MOVIELENS_DIR:-data/raw/ml-latest-small}"
MOVIELENS_CATALOG="${MOVIELENS_CATALOG:-data/processed/movie_catalog_enriched.parquet}"
LETTERBOXD_DIR="${LETTERBOXD_DIR:-data/processed/letterboxd}"
LETTERBOXD_CATALOG="${LETTERBOXD_CATALOG:-data/processed/letterboxd/movie_catalog_enriched.parquet}"

log "CONTENT_BACKEND: $CONTENT_BACKEND"
log "SBERT_MODEL: $SBERT_MODEL"
log "DEVICE: $DEVICE"
log "EPOCHS: $EPOCHS"
log "BATCH_SIZE: $BATCH_SIZE"
log "DIM: $DIM"
log "GRID_STEP: $GRID_STEP"

if [[ "$DEVICE" == cuda* ]]; then
  run_step check_cuda "$PYTHON_BIN" -u -c '
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use DEVICE=cpu or enable Kaggle GPU.")
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0))
'
fi

if [[ "$CONTENT_BACKEND" == "sbert" || "$CONTENT_BACKEND" == "auto" ]]; then
  run_step check_sentence_transformers "$PYTHON_BIN" -u -c '
import sentence_transformers

print("sentence_transformers:", sentence_transformers.__version__)
'
fi

if [[ "$RUN_MOVIELENS" == "1" ]]; then
  run_step train_movielens_deadline "$PYTHON_BIN" -u scripts/train.py \
    --raw-dir "$MOVIELENS_DIR" \
    --enriched-catalog "$MOVIELENS_CATALOG" \
    --artifacts-dir artifacts/movielens_deadline \
    --content-backend "$CONTENT_BACKEND" \
    --sbert-model "$SBERT_MODEL" \
    --train-lightgcn \
    --train-two-tower \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hybrid-grid-step "$GRID_STEP" \
    --min-rating "$MIN_RATING"
fi

if [[ "$RUN_LETTERBOXD" == "1" ]]; then
  run_step train_letterboxd_deadline "$PYTHON_BIN" -u scripts/train.py \
    --raw-dir "$LETTERBOXD_DIR" \
    --enriched-catalog "$LETTERBOXD_CATALOG" \
    --artifacts-dir artifacts/letterboxd_deadline \
    --content-backend "$CONTENT_BACKEND" \
    --sbert-model "$SBERT_MODEL" \
    --train-lightgcn \
    --train-two-tower \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --hybrid-grid-step "$GRID_STEP" \
    --min-rating "$MIN_RATING"
fi

run_step audit_deadline_artifacts "$PYTHON_BIN" -u scripts/audit_artifacts.py \
  --target movielens_deadline=artifacts/movielens_deadline \
  --target letterboxd_deadline=artifacts/letterboxd_deadline

if command -v zip >/dev/null 2>&1; then
  ZIP_PATHS=()
  for path in artifacts/movielens_deadline artifacts/letterboxd_deadline "$LOG_DIR"; do
    if [[ -e "$path" ]]; then
      ZIP_PATHS+=("$path")
    else
      log "SKIP zip missing path: $path"
    fi
  done
  if [[ "${#ZIP_PATHS[@]}" -gt 0 ]]; then
    run_step zip_deadline_outputs zip -qr deadline_artifacts.zip "${ZIP_PATHS[@]}"
  fi
  log "Created deadline_artifacts.zip"
fi

log "Deadline training done"
