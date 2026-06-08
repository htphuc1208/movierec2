# Comparison Summary

Ranking metrics use train-item masking and Top-10 evaluation on the test split.

## movielens

| Model | Precision@10 | Recall@10 | NDCG@10 | MRR | Seconds |
|---|---:|---:|---:|---:|---:|
| ease | 0.0299 | 0.0612 | 0.0506 | 0.0898 | 11.54 |
| svd_ranking | 0.0250 | 0.0604 | 0.0454 | 0.0727 | 0.17 |
| item_knn_cosine | 0.0275 | 0.0554 | 0.0438 | 0.0736 | 0.20 |
| user_knn_cosine | 0.0255 | 0.0554 | 0.0435 | 0.0736 | 0.08 |
| hybrid_weighted_no_popularity | 0.0235 | 0.0546 | 0.0426 | 0.0698 | 2.90 |
| hybrid_weighted_full | 0.0201 | 0.0432 | 0.0377 | 0.0688 | 0.34 |
| hybrid_ranker_full | 0.0204 | 0.0439 | 0.0366 | 0.0655 | 4.72 |
| popularity_only | 0.0194 | 0.0397 | 0.0365 | 0.0677 | 0.02 |
| lightgcn_only | 0.0184 | 0.0390 | 0.0359 | 0.0667 | 17.41 |
| hybrid_ranker_no_popularity | 0.0204 | 0.0428 | 0.0354 | 0.0615 | 5.23 |
| hybrid_no_tmdb | 0.0192 | 0.0464 | 0.0329 | 0.0541 | 2.52 |
| learned_two_tower | 0.0117 | 0.0265 | 0.0204 | 0.0356 | 14.43 |
| tfidf_only | 0.0063 | 0.0171 | 0.0113 | 0.0184 | 0.13 |
| random | 0.0013 | 0.0018 | 0.0030 | 0.0084 | 0.03 |
| bpr_mf | 0.0013 | 0.0033 | 0.0025 | 0.0041 | 19.41 |
