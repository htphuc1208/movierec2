| model | source | ndcg@10 | mrr@10 | recall@10 | precision@10 | rmse | artifact_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid-recbolelightgcn-tfidf | tuned_hybrid | 0.02593016446635522 | 0.038328772121875576 | 0.033970078452837074 | 0.009195402298850575 | 1.740047574043274 | artifacts/recommender/letterboxd-full/hybrid-recbolelightgcn-tfidf |
| svd-pytorch | pytorch_svd | 0.02413361425407942 | 0.03344736361977741 | 0.03369557645419714 | 0.009425287356321841 | 0.7890205979347229 | artifacts/recommender/letterboxd-full/svd |
| hybrid-lightgcn-tfidf | tuned_hybrid | 0.02391205695754929 | 0.032112753147235905 | 0.03348841452289728 | 0.00781609195402299 | 1.5272088050842285 | artifacts/recommender/letterboxd-full/hybrid-lightgcn-tfidf |
| lightgcn-pytorch | pytorch_lightgcn | 0.02319415813944322 | 0.03708721036307243 | 0.028767560664112388 | 0.010344827586206896 |  | artifacts/recommender/letterboxd-full/lightgcn |
| recbole-LightGCN | recbole | 0.0221 | 0.0329 | 0.031 | 0.0108 |  | artifacts/recommender/letterboxd-full/recbole-lightgcn |
| hybrid-svd-tfidf | tuned_hybrid | 0.02191123137688706 | 0.019797482211275315 | 0.03902124695228143 | 0.00735632183908046 | 0.7832956314086914 | artifacts/recommender/letterboxd-full/hybrid-svd-tfidf |
| hybrid-pytorch-svd-tfidf | runtime_fit | 0.02171540937987829 | 0.01743659916073709 | 0.04097526994078718 | 0.00735632183908046 | 0.7831368446350098 | artifacts/recommender/letterboxd-full/baseline |
| hybrid-bpr-tfidf | tuned_hybrid | 0.021593103579572975 | 0.031360153256704985 | 0.028137201240649515 | 0.008505747126436782 | 1.8431804180145264 | artifacts/recommender/letterboxd-full/hybrid-bpr-tfidf |
| hybrid-twotower-tfidf | tuned_hybrid | 0.018458293779098543 | 0.02121510673234811 | 0.028071519795657723 | 0.007126436781609196 | 1.6981083154678345 | artifacts/recommender/letterboxd-full/hybrid-twotower-tfidf |
| recbole-BPR | recbole | 0.0184 | 0.0243 | 0.0262 | 0.0087 |  | artifacts/recommender/letterboxd-full/recbole-bpr |
| recbole-ItemKNN | recbole | 0.0143 | 0.0212 | 0.023 | 0.0067 |  |  |
| recbole-Pop | recbole | 0.0124 | 0.017 | 0.0207 | 0.0071 |  |  |
| two-tower-tfidf | pytorch_two_tower | 0.003779901207639854 | 0.006675789089582193 | 0.00553639846743295 | 0.0025287356321839084 |  | artifacts/recommender/letterboxd-full/two-tower |
