| model | source | ndcg@10 | mrr@10 | recall@10 | precision@10 | rmse | artifact_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-bpr-tfidf | tuned_hybrid | 0.057941185294224415 | 0.10066348643476304 | 0.06736923098447234 | 0.03687943262411348 | 1.6615620851516724 | artifacts/recommender/ml-latest-small/hybrid-bpr-tfidf |
| hybrid-twotower-tfidf | tuned_hybrid | 0.057229086388950974 | 0.09770277496341326 | 0.06640179650910481 | 0.036702127659574466 | 1.9540351629257202 | artifacts/recommender/ml-latest-small/hybrid-twotower-tfidf |
| recbole-ItemKNN | recbole | 0.0524 | 0.0978 | 0.0597 | 0.0329 |  |  |
| hybrid-recbolelightgcn-tfidf | tuned_hybrid | 0.051644008618287815 | 0.09066475289879544 | 0.05756631505036966 | 0.03351063829787235 | 1.796669363975525 | artifacts/recommender/ml-latest-small/hybrid-recbolelightgcn-tfidf |
| hybrid-lightgcn-tfidf | tuned_hybrid | 0.050987718476633415 | 0.09149850838680626 | 0.055362539699285056 | 0.03280141843971631 | 1.978361964225769 | artifacts/recommender/ml-latest-small/hybrid-lightgcn-tfidf |
| hybrid-svd-tfidf | tuned_hybrid | 0.05084923800457362 | 0.08782224473713836 | 0.05569108757921667 | 0.03280141843971631 | 0.9043042659759521 | artifacts/recommender/ml-latest-small/hybrid-svd-tfidf |
| recbole-BPR | recbole | 0.0476 | 0.0878 | 0.0565 | 0.0276 |  | artifacts/recommender/ml-latest-small/recbole-bpr |
| lightgcn-pytorch | pytorch_lightgcn | 0.04471863904219621 | 0.08541877744005404 | 0.04586701778455716 | 0.026418439716312057 |  | artifacts/recommender/ml-latest-small/lightgcn |
| hybrid-pytorch-svd-tfidf | runtime_fit | 0.044049261405181075 | 0.08132950579759089 | 0.04752550885880182 | 0.02695035460992908 | 0.8999696373939514 | artifacts/recommender/ml-latest-small/baseline |
| recbole-LightGCN | recbole | 0.0408 | 0.0786 | 0.0435 | 0.0228 |  | artifacts/recommender/ml-latest-small/recbole-lightgcn |
| recbole-Pop | recbole | 0.031 | 0.0591 | 0.0376 | 0.0202 |  |  |
| two-tower-tfidf | pytorch_two_tower | 0.01798178953409427 | 0.04116219182708545 | 0.016009488302659424 | 0.01152482269503546 |  | artifacts/recommender/ml-latest-small/two-tower |
| svd-pytorch | pytorch_svd | 0.007209960814059993 | 0.015052628616458402 | 0.008374384901718287 | 0.00549645390070922 | 0.8921972513198853 | artifacts/recommender/ml-latest-small/svd |
