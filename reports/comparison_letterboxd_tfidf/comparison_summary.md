# Comparison Summary

Ranking metrics use train-item masking and Top-10 evaluation on the test split.

## letterboxd

Split stats:

- users=8985, items=7211, train=160843, val=20669, test=20842
- sparse_users=2342, warm_users=6643, long_tail_items=3564, head_items=1827

| Group | Model | Precision@10 | Recall@10 | NDCG@10 | MRR | Sparse NDCG@10 | Tail NDCG@10 | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baselines | ease | 0.0480 | 0.1940 | 0.1376 | 0.1742 | 0.0880 | 0.0000 | 14.98 |
| pdf_clean_and_ablation | hybrid_ranker_full | 0.0402 | 0.1732 | 0.1219 | 0.1517 | 0.1207 | 0.0098 | 162.94 |
| pdf_clean_and_ablation | hybrid_ranker_no_popularity | 0.0401 | 0.1719 | 0.1215 | 0.1512 | 0.1212 | 0.0100 | 157.71 |
| pdf_clean_and_ablation | hybrid_weighted_no_popularity | 0.0401 | 0.1711 | 0.1190 | 0.1487 | 0.1053 | 0.0046 | 35.39 |
| pdf_clean_and_ablation | hybrid_weighted_full | 0.0401 | 0.1711 | 0.1190 | 0.1487 | 0.1053 | 0.0046 | 81.98 |
| baselines | user_knn_cosine | 0.0409 | 0.1671 | 0.1155 | 0.1469 | 0.0726 | 0.0000 | 2.18 |
| baselines | item_knn_cosine | 0.0402 | 0.1625 | 0.1148 | 0.1483 | 0.0672 | 0.0069 | 1.53 |
| pdf_clean_and_ablation | lightgcn_only | 0.0356 | 0.1441 | 0.0959 | 0.1211 | 0.0538 | 0.0009 | 5585.92 |
| pdf_clean_and_ablation | hybrid_no_tmdb | 0.0356 | 0.1441 | 0.0959 | 0.1211 | 0.0538 | 0.0009 | 33.70 |
| baselines | svd_ranking | 0.0293 | 0.1139 | 0.0809 | 0.1091 | 0.0493 | 0.0000 | 2.50 |
| baselines | bpr_mf | 0.0287 | 0.1122 | 0.0767 | 0.0999 | 0.0417 | 0.0004 | 2516.90 |
| baselines | popularity_only | 0.0199 | 0.0831 | 0.0525 | 0.0636 | 0.0317 | 0.0000 | 1.16 |
| pdf_clean_and_ablation | tfidf_only | 0.0074 | 0.0526 | 0.0339 | 0.0318 | 0.0997 | 0.0323 | 2.65 |
| pdf_clean_and_ablation | learned_two_tower | 0.0125 | 0.0498 | 0.0299 | 0.0372 | 0.0165 | 0.0000 | 2283.40 |
| baselines | random | 0.0006 | 0.0022 | 0.0010 | 0.0010 | 0.0000 | 0.0009 | 1.06 |
