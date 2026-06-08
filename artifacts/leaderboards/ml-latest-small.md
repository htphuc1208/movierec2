| model | source | ndcg@10 | warm_ndcg@10 | cold_ndcg@10 | mrr@10 | recall@10 | precision@10 | rmse | artifact_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-bpr-tfidf | tuned_hybrid | 0.05441215166541153 | 0.05471038375188991 | 0.0007888593774124736 | 0.09638917032534053 | 0.060331839007955704 | 0.0375886524822695 | 1.6976710557937622 | artifacts/recommender/ml-latest-small/hybrid-bpr-tfidf |
| recbole-ItemKNN | recbole | 0.0524 |  |  | 0.0978 | 0.0597 | 0.0329 |  |  |
| hybrid-lightgcn-tfidf | tuned_hybrid | 0.051144066655759414 | 0.05155265123368959 | 0.0 | 0.09245328154902623 | 0.058782449709003384 | 0.03439716312056738 | 1.4535467624664307 | artifacts/recommender/ml-latest-small/hybrid-lightgcn-tfidf |
| hybrid-recbolelightgcn-tfidf | tuned_hybrid | 0.04964578417701834 | 0.050051198818809366 | 0.0 | 0.09081461780929866 | 0.05578432059132007 | 0.03333333333333333 | 1.5666486024856567 | artifacts/recommender/ml-latest-small/hybrid-recbolelightgcn-tfidf |
| hybrid-twotower-tfidf | tuned_hybrid | 0.04907402526989376 | 0.04941059925198809 | 0.0 | 0.09163148711021052 | 0.05285194203839269 | 0.031737588652482274 | 1.4845631122589111 | artifacts/recommender/ml-latest-small/hybrid-twotower-tfidf |
| recbole-BPR | recbole | 0.0476 |  |  | 0.0878 | 0.0565 | 0.0276 |  | artifacts/recommender/ml-latest-small/recbole-bpr |
| hybrid-svd-tfidf | tuned_hybrid | 0.04485573117914506 | 0.04525301838434683 | 0.0 | 0.08090875830237532 | 0.047073335024590775 | 0.03102836879432624 | 0.9043042659759521 | artifacts/recommender/ml-latest-small/hybrid-svd-tfidf |
| lightgcn-pytorch | pytorch_lightgcn | 0.0447071259741433 | 0.044951973386755086 | 0.0 | 0.08535826860294946 | 0.04583478051827734 | 0.026418439716312057 |  | artifacts/recommender/ml-latest-small/lightgcn |
| recbole-LightGCN | recbole | 0.0408 |  |  | 0.0786 | 0.0435 | 0.0228 |  | artifacts/recommender/ml-latest-small/recbole-lightgcn |
| hybrid-pytorch-svd-tfidf | runtime_fit | 0.033840030387801395 | 0.034112970756946116 | 0.0 | 0.06826381852977598 | 0.03322790441518959 | 0.02375886524822695 | 0.9903233051300049 | artifacts/recommender/ml-latest-small/baseline |
| hybrid-baseline | runtime_fit | 0.033840030387801395 | 0.034112970756946116 | 0.0 | 0.06826381852977598 | 0.03322790441518959 | 0.02375886524822695 | 0.9903233051300049 | artifacts/recommender/ml-latest-small/baseline |
| recbole-Pop | recbole | 0.031 |  |  | 0.0591 | 0.0376 | 0.0202 |  |  |
| content-tfidf | content_baseline | 0.02138752035043007 | 0.021820957654541773 | 0.0009155173939959246 | 0.03854905437352246 | 0.025430155231707878 | 0.013297872340425532 |  | artifacts/recommender/ml-latest-small/content |
| two-tower-tfidf | pytorch_two_tower | 0.017320780748322807 | 0.01740902869247414 | 0.0 | 0.03435916919959473 | 0.01875156065977906 | 0.010815602836879434 |  | artifacts/recommender/ml-latest-small/two-tower |
| svd-pytorch | pytorch_svd | 0.007209960814059993 | 0.007353428080451418 | 0.0 | 0.015052628616458402 | 0.008374384901718287 | 0.00549645390070922 | 0.8921972513198853 | artifacts/recommender/ml-latest-small/svd |
