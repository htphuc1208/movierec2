# SVD Baseline Notes

## What Was Researched

For explicit MovieLens ratings, the stronger "SVD" baseline is biased matrix factorization, not raw linear-algebra SVD on a zero-filled matrix. This is the family described by Koren, Bell, and Volinsky for recommender systems and exposed as `SVD`/`SVD++` in common recommender libraries.

## Current Repo Comparison

The original `models/recommender.py` baseline used `sklearn.decomposition.TruncatedSVD` inside the default hybrid demo. That was useful because it was lightweight and had no PyTorch dependency, but it decomposed a dense user-item matrix where missing ratings were represented as zeros.

`models/SVD.py` implements Funk-SVD style biased matrix factorization:

```text
prediction = global_mean + user_bias + item_bias + dot(user_embedding, item_embedding)
```

This is a better rating-prediction baseline because it trains only on observed ratings and learns user/item biases.

The default `HybridMovieRecommender` now uses the same biased matrix-factorization formula in a small numpy trainer so the API path does not require PyTorch. The separate `scripts/train_svd.py` pipeline still uses `models/SVD.py` for independent PyTorch experiments and artifact export.

## Independent Pipeline

`scripts/train_svd.py` trains `models/SVD.py` independently from the API and from `scripts/train_baseline.py`.

It provides:

- timestamp-based train/validation/test split
- stable `userId` and `movieId` mappings
- AdamW training with MSE loss
- early stopping on validation RMSE
- RMSE, MAE, Precision@K, Recall@K, NDCG@K, MRR@K
- artifact saving under `artifacts/`

## Trial Results Before Core Integration

On `data/ml-latest-small` with 64 factors and early stopping:

```text
Funk-SVD test_rmse: 0.9478
Funk-SVD test_mae: 0.7318
Funk-SVD precision@10: 0.0119
Funk-SVD recall@10: 0.0216
Funk-SVD ndcg@10: 0.0227
Funk-SVD mrr@10: 0.0486
```

The default hybrid baseline on the same data:

```text
Hybrid baseline rmse: 2.1629
Hybrid baseline precision@10: 0.0381
Hybrid baseline recall@10: 0.0717
Hybrid baseline ndcg@10: 0.0593
Hybrid baseline mrr@10: 0.0992
```

Interpretation: Funk-SVD is much better for rating prediction. The current hybrid baseline ranks better in this quick run because it directly uses content and popularity signals. The next step is to integrate Funk-SVD scores into the hybrid scorer and tune weights on validation data.

## Trial Results After Core Integration

After replacing the default collaborative component with numpy Funk-SVD, `scripts/train_baseline.py` on `data/ml-latest-small` reports:

```text
rmse: 0.9017
precision@10: 0.0248
recall@10: 0.0422
ndcg@10: 0.0356
mrr@10: 0.0612
```

Interpretation: the core baseline now has much better rating prediction. Top-K ranking dropped relative to the old `TruncatedSVD + TF-IDF + popularity` weights, so the next ranking-oriented step should tune hybrid weights on validation data or replace the ranking component with LightGCN/BPR.
