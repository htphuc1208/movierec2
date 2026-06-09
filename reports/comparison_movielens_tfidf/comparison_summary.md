# Comparison Summary

Ranking metrics use train-item masking and Top-10 evaluation on the test split.

## movielens

Split stats:

- users=609, items=6298, train=38833, val=4872, test=4875
- sparse_users=165, warm_users=444, long_tail_items=2950, head_items=1443

| Group | Model | Precision@10 | Recall@10 | NDCG@10 | MRR | Sparse NDCG@10 | Tail NDCG@10 | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baselines | ease | 0.0299 | 0.0612 | 0.0506 | 0.0898 | 0.0430 | 0.0000 | 16.30 |
| baselines | svd_ranking | 0.0250 | 0.0604 | 0.0454 | 0.0727 | 0.0454 | 0.0000 | 0.87 |
| baselines | item_knn_cosine | 0.0275 | 0.0554 | 0.0438 | 0.0736 | 0.0394 | 0.0007 | 0.38 |
| pdf_clean_and_ablation | hybrid_ranker_full | 0.0248 | 0.0515 | 0.0437 | 0.0743 | 0.0507 | 0.0000 | 18.03 |
| baselines | user_knn_cosine | 0.0255 | 0.0554 | 0.0435 | 0.0736 | 0.0437 | 0.0000 | 0.32 |
| pdf_clean_and_ablation | hybrid_weighted_no_popularity | 0.0232 | 0.0565 | 0.0429 | 0.0727 | 0.0489 | 0.0000 | 4.03 |
| pdf_clean_and_ablation | hybrid_weighted_full | 0.0232 | 0.0565 | 0.0429 | 0.0727 | 0.0489 | 0.0000 | 8.55 |
| pdf_clean_and_ablation | hybrid_no_tmdb | 0.0232 | 0.0542 | 0.0406 | 0.0664 | 0.0479 | 0.0000 | 3.79 |
| pdf_clean_and_ablation | hybrid_ranker_no_popularity | 0.0245 | 0.0523 | 0.0403 | 0.0659 | 0.0415 | 0.0000 | 18.62 |
| pdf_clean_and_ablation | lightgcn_only | 0.0220 | 0.0458 | 0.0394 | 0.0687 | 0.0465 | 0.0000 | 411.34 |
| baselines | bpr_mf | 0.0242 | 0.0476 | 0.0388 | 0.0696 | 0.0364 | 0.0000 | 147.72 |
| baselines | popularity_only | 0.0194 | 0.0397 | 0.0365 | 0.0677 | 0.0409 | 0.0000 | 0.10 |
| pdf_clean_and_ablation | learned_two_tower | 0.0137 | 0.0324 | 0.0272 | 0.0478 | 0.0274 | 0.0000 | 137.24 |
| pdf_clean_and_ablation | tfidf_only | 0.0063 | 0.0171 | 0.0113 | 0.0184 | 0.0153 | 0.0074 | 0.37 |
| baselines | random | 0.0013 | 0.0018 | 0.0030 | 0.0084 | 0.0000 | 0.0000 | 0.13 |
