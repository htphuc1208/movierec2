#!/usr/bin/env bash
set -euo pipefail

# Focused L40S runner for the remaining strong-ranker artifacts.
# Defaults to Letterboxd because it is the larger, stronger target; set
# RUN_MOVIELENS=1 to also fill artifacts/movielens_strong.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RECOMMENDER_VERBOSE_TRAIN="${RECOMMENDER_VERBOSE_TRAIN:-1}"

LOG_DIR="${LOG_DIR:-logs/l40s_strong_$(date +%Y%m%d_%H%M%S)}"
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
  run_step install_requirements "$PYTHON_BIN" -m pip install -q --upgrade -r requirements.txt
  run_step uninstall_unneeded_torch_extras "$PYTHON_BIN" -m pip uninstall -y torchcodec torchvision torchaudio
  if [[ -f requirements-optional.txt ]]; then
    while IFS= read -r optional_pkg || [[ -n "$optional_pkg" ]]; do
      [[ -z "$optional_pkg" || "$optional_pkg" =~ ^# ]] && continue
      optional_name="install_optional_${optional_pkg//[^A-Za-z0-9_]/_}"
      optional_log="$LOG_DIR/${optional_name}.log"
      log "START $optional_name"
      log "LOG $optional_log"
      log "CMD $PYTHON_BIN -m pip install -q --upgrade $optional_pkg"
      set +e
      "$PYTHON_BIN" -m pip install -q --upgrade "$optional_pkg" 2>&1 | tee "$optional_log"
      optional_status=${PIPESTATUS[0]}
      set -e
      if [[ "$optional_status" -ne 0 ]]; then
        log "WARNING optional dependency failed: $optional_pkg status=$optional_status"
      else
        log "DONE $optional_name"
      fi
    done < requirements-optional.txt
  fi
fi

DEVICE="${DEVICE:-cuda}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
CONTENT_BACKEND="${CONTENT_BACKEND:-sbert}"
SBERT_MODEL="${SBERT_MODEL:-sentence-transformers/all-mpnet-base-v2}"
RANKER="${RANKER:-lightgbm}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
DIM="${DIM:-128}"
MAX_EASE_ITEMS="${MAX_EASE_ITEMS:-5000}"
MAX_RANKER_SAMPLES="${MAX_RANKER_SAMPLES:-500000}"
MIN_RATING="${MIN_RATING:-4.0}"

RUN_MOVIELENS="${RUN_MOVIELENS:-0}"
RUN_LETTERBOXD="${RUN_LETTERBOXD:-1}"

MOVIELENS_DIR="${MOVIELENS_DIR:-data/raw/ml-latest-small}"
MOVIELENS_CATALOG="${MOVIELENS_CATALOG:-data/processed/movie_catalog_enriched.parquet}"
LETTERBOXD_DIR="${LETTERBOXD_DIR:-data/processed/letterboxd}"
LETTERBOXD_CATALOG="${LETTERBOXD_CATALOG:-data/processed/letterboxd/movie_catalog_enriched.parquet}"

log "DEVICE: $DEVICE"
log "CONTENT_BACKEND: $CONTENT_BACKEND"
log "SBERT_MODEL: $SBERT_MODEL"
log "RANKER: $RANKER"
log "EPOCHS: $EPOCHS"
log "BATCH_SIZE: $BATCH_SIZE"
log "DIM: $DIM"
log "MAX_EASE_ITEMS: $MAX_EASE_ITEMS"
log "MAX_RANKER_SAMPLES: $MAX_RANKER_SAMPLES"
log "RUN_MOVIELENS: $RUN_MOVIELENS"
log "RUN_LETTERBOXD: $RUN_LETTERBOXD"

if [[ "$DEVICE" == cuda* || "$REQUIRE_CUDA" == "1" ]]; then
  run_step check_cuda "$PYTHON_BIN" -u -c '
import torch

print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run this script inside the L40S environment or set REQUIRE_CUDA=0 DEVICE=cpu for a smoke run.")
print("device_count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"device_{idx}:", torch.cuda.get_device_name(idx))
name = torch.cuda.get_device_name(0)
if "L40" not in name:
    print(f"WARNING: expected an L40/L40S-class GPU, got {name!r}")
probe = torch.tensor([1.0], device="cuda")
print("cuda_probe:", float((probe + 1).cpu()[0]))
'
fi

if [[ "$CONTENT_BACKEND" == "sbert" || "$CONTENT_BACKEND" == "auto" ]]; then
  run_step check_sentence_transformers "$PYTHON_BIN" -u -c '
import sentence_transformers
import transformers

print("sentence_transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)
major = int(sentence_transformers.__version__.split(".", 1)[0])
if major >= 4:
    raise SystemExit("sentence-transformers must be <4.0 for this project runtime.")
'
fi

if [[ "$RUN_MOVIELENS" == "1" ]]; then
  run_step train_movielens_strong "$PYTHON_BIN" -u scripts/train_strong_hybrid.py \
    --dataset movielens \
    --raw-dir "$MOVIELENS_DIR" \
    --enriched-catalog "$MOVIELENS_CATALOG" \
    --artifacts-dir artifacts/movielens_strong \
    --content-backend "$CONTENT_BACKEND" \
    --sbert-model "$SBERT_MODEL" \
    --ranker "$RANKER" \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --lightgcn-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max-ease-items "$MAX_EASE_ITEMS" \
    --max-ranker-samples "$MAX_RANKER_SAMPLES" \
    --min-rating "$MIN_RATING"
fi

if [[ "$RUN_LETTERBOXD" == "1" ]]; then
  run_step train_letterboxd_strong "$PYTHON_BIN" -u scripts/train_strong_hybrid.py \
    --dataset letterboxd \
    --raw-dir "$LETTERBOXD_DIR" \
    --enriched-catalog "$LETTERBOXD_CATALOG" \
    --artifacts-dir artifacts/letterboxd_strong \
    --content-backend "$CONTENT_BACKEND" \
    --sbert-model "$SBERT_MODEL" \
    --ranker "$RANKER" \
    --lightgcn-dim "$DIM" \
    --lightgcn-layers 3 \
    --lightgcn-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --max-ease-items "$MAX_EASE_ITEMS" \
    --max-ranker-samples "$MAX_RANKER_SAMPLES" \
    --min-rating "$MIN_RATING"
fi

AUDIT_TARGETS=()
if [[ "$RUN_MOVIELENS" == "1" ]]; then
  AUDIT_TARGETS+=(--target movielens_strong=artifacts/movielens_strong)
fi
if [[ "$RUN_LETTERBOXD" == "1" ]]; then
  AUDIT_TARGETS+=(--target letterboxd_strong=artifacts/letterboxd_strong)
fi

if [[ "${#AUDIT_TARGETS[@]}" -gt 0 ]]; then
  run_step audit_strong_artifacts "$PYTHON_BIN" -u scripts/audit_artifacts.py "${AUDIT_TARGETS[@]}"
fi

if command -v zip >/dev/null 2>&1; then
  ZIP_PATHS=("$LOG_DIR")
  for path in artifacts/movielens_strong artifacts/letterboxd_strong; do
    if [[ -e "$path" ]]; then
      ZIP_PATHS+=("$path")
    fi
  done
  run_step zip_l40s_strong_outputs zip -qr l40s_strong_outputs.zip "${ZIP_PATHS[@]}"
  log "Created l40s_strong_outputs.zip"
fi

log "L40S strong training done"
