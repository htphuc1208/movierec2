# Kaggle Training Guide

This guide is for the PDF-aligned experiment:

1. Native PyTorch LightGCN on MovieLens `ml-latest`
2. WARP ranking loss
3. SBERT content vectors
4. Tuned hybrid scoring: LightGCN + SBERT + popularity

## Setup

Use a GPU runtime when available. Install the normal project dependencies plus the ML extras:

```bash
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

If internet is enabled, download MovieLens latest full:

```bash
python scripts/download_movielens.py --variant ml-latest --output-dir data
```

If internet is disabled, add the MovieLens latest full dataset to the Kaggle notebook and place or symlink its folder as:

```text
data/ml-latest/
```

Expected files include `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `genome-scores.csv`, and `genome-tags.csv`.

## Train LightGCN With WARP

Start with this full experiment when GPU/RAM allow it:

```bash
python scripts/train_lightgcn.py \
  --data-dir data/ml-latest \
  --dataset-name ml-latest \
  --epochs 20 \
  --batch-size 8192 \
  --factors 64 \
  --layers 3 \
  --loss warp \
  --warp-margin 1.0 \
  --warp-max-trials 20 \
  --artifact-path artifacts/checkpoints/ml-latest/lightgcn-warp.pt \
  --recommender-artifact-dir artifacts/recommender/ml-latest/lightgcn-warp \
  --top-k 10
```

If the run is too slow for the notebook session, use sampled training/evaluation. The artifact config records these limits:

```bash
python scripts/train_lightgcn.py \
  --data-dir data/ml-latest \
  --dataset-name ml-latest \
  --epochs 5 \
  --batch-size 8192 \
  --factors 32 \
  --layers 2 \
  --loss warp \
  --warp-max-trials 10 \
  --max-train-pairs 1000000 \
  --eval-user-limit 5000 \
  --artifact-path artifacts/checkpoints/ml-latest/lightgcn-warp-sampled.pt \
  --recommender-artifact-dir artifacts/recommender/ml-latest/lightgcn-warp-sampled \
  --top-k 10
```

## Tune LightGCN + SBERT Hybrid

After LightGCN finishes, tune hybrid weights using SBERT content vectors:

```bash
python scripts/tune_hybrid.py \
  --data-dir data/ml-latest \
  --dataset-name ml-latest \
  --cf-model LightGCN \
  --cf-artifact-dir artifacts/recommender/ml-latest/lightgcn-warp \
  --content-backend sbert \
  --output-dir artifacts/recommender/ml-latest/hybrid-lightgcn-sbert \
  --eval-user-limit 5000 \
  --top-k 10
```

For the sampled LightGCN artifact, change `--cf-artifact-dir`:

```bash
python scripts/tune_hybrid.py \
  --data-dir data/ml-latest \
  --dataset-name ml-latest \
  --cf-model LightGCN \
  --cf-artifact-dir artifacts/recommender/ml-latest/lightgcn-warp-sampled \
  --content-backend sbert \
  --output-dir artifacts/recommender/ml-latest/hybrid-lightgcn-sbert-sampled \
  --eval-user-limit 5000 \
  --top-k 10
```

## Run API With The Trained Artifact

```bash
MOVIEREC_DATA_DIR=data/ml-latest \
MOVIEREC_ARTIFACT_DIR=artifacts/recommender/ml-latest/hybrid-lightgcn-sbert \
uvicorn main:app --host 0.0.0.0 --port 8000
```

The artifact should contain:

```text
manifest.json
collaborative.npz
content.npz
```

`content.npz` stores precomputed SBERT item vectors, so the API does not need to encode all movie texts at startup.
