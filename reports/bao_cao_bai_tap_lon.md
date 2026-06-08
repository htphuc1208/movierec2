# Báo cáo bài tập lớn: Hệ thống gợi ý phim Hybrid sử dụng Collaborative Filtering, Content-based Learning và RAG Chatbot

## 1. Thông tin đề tài

**Tên đề tài:** Xây dựng hệ thống gợi ý phim Hybrid từ dữ liệu MovieLens, Letterboxd và metadata TMDb.

**Loại bài toán:** Hệ thống gợi ý cá nhân hóa trong miền phim ảnh.

**Mục tiêu chính:** Xây dựng một hệ thống học máy có khả năng đề xuất danh sách phim phù hợp cho từng người dùng hoặc cho một phiên xem mới, đồng thời cung cấp giao diện demo, API inference, dashboard phân tích dữ liệu và chatbot tư vấn phim.

**Ý tưởng tổng quát:** Trong thực tế, người dùng thường gặp tình trạng quá tải lựa chọn khi có quá nhiều phim trên các nền tảng như Netflix, IMDb, Letterboxd hoặc các dịch vụ streaming. Một hệ thống gợi ý cần khai thác cả hành vi người dùng, như lịch sử đánh giá phim, và thông tin nội dung phim, như thể loại, mô tả, đạo diễn, diễn viên, từ khóa, độ phổ biến. Vì vậy, đồ án xây dựng một hệ thống hybrid kết hợp:

- Collaborative Filtering: học từ quan hệ giữa người dùng và phim.
- Content-based Recommendation: học từ metadata và mô tả nội dung phim.
- Hybrid Scoring: kết hợp nhiều nguồn điểm để xếp hạng phim.
- Model comparison: so sánh với các baseline phổ biến.
- RAG Chatbot: truy xuất phim liên quan từ catalog và trả lời bằng tiếng Việt.

## 2. Mô tả bài toán thực tế

### 2.1. Bối cảnh

Một người dùng muốn tìm phim để xem nhưng không muốn duyệt thủ công hàng nghìn phim. Người dùng có thể rơi vào một trong hai kịch bản:

1. **User đã có lịch sử đánh giá:** hệ thống biết user từng thích phim nào và có thể cá nhân hóa sâu.
2. **User mới hoặc phiên xem mới:** hệ thống chưa biết user là ai nhưng có thể dựa trên một vài phim người dùng chọn trong phiên hiện tại.

Ngoài ra, người dùng có thể hỏi bằng ngôn ngữ tự nhiên, ví dụ:

- "Tôi muốn xem phim khoa học viễn tưởng về không gian."
- "Có phim nào giống The Dark Knight không?"
- "Gợi ý phim drama cảm động, có diễn viên nổi tiếng."

### 2.2. Bài toán cần giải quyết

Với một user hoặc một session context, hệ thống cần trả về danh sách Top-K phim phù hợp nhất.

**Đầu vào của bài toán:**

- `user_id`: ID người dùng nếu có.
- `session_context`: danh sách phim người dùng chọn trong phiên hiện tại.
- `top_k`: số lượng phim cần gợi ý.
- `model_name`: chế độ gợi ý, ví dụ `hybrid`, `lightgcn`, `content`, `popularity`, `strong`.
- Với chatbot: câu hỏi tự nhiên của người dùng.

**Đầu ra của bài toán:**

Danh sách phim được xếp hạng, mỗi phim gồm:

- `movie_id`
- `tmdb_id`
- `title`
- `score`
- `poster_url`
- `genres`
- `overview`
- `director`
- `cast`
- `explanation_tags`

Ví dụ response của API:

```json
{
  "recommendations": [
    {
      "movie_id": 2571,
      "tmdb_id": 603,
      "title": "The Matrix",
      "score": 0.94,
      "poster_url": "...",
      "explanation_tags": ["phù hợp lịch sử đánh giá", "cùng thể loại: Action, Sci-Fi"]
    }
  ]
}
```

### 2.3. Yêu cầu hệ thống

Hệ thống cần đáp ứng các yêu cầu sau:

- Huấn luyện được mô hình từ dữ liệu rating/interactions.
- Kết hợp được tín hiệu collaborative và content.
- Có khả năng đánh giá bằng các ranking metrics.
- Có thể chạy inference qua API mà không train lại.
- Có giao diện demo dễ sử dụng.
- Có chức năng tìm kiếm phim, xem chi tiết, xem phim tương tự, chấm rating.
- Có dashboard EDA và visualization phục vụ phân tích/báo cáo.
- Có chatbot truy vấn catalog phim bằng ngôn ngữ tự nhiên.

## 3. Dữ liệu sử dụng

Đồ án sử dụng ba nhóm dữ liệu chính.

### 3.1. MovieLens

MovieLens là tập dữ liệu chuẩn cho bài toán recommender system.

**Nguồn sử dụng trong repo:** `data/raw/ml-latest-small`

Các file chính:

- `ratings.csv`: tương tác người dùng-phim.
- `movies.csv`: thông tin phim cơ bản.
- `links.csv`: mapping sang IMDb/TMDb.

Thông tin hiện tại của `ml-latest-small` trong dự án:

| Thuộc tính | Giá trị |
|---|---:|
| Số interactions gốc | 100,836 |
| Số users gốc | 610 |
| Số movies gốc | 9,724 |
| Positive interactions với `rating >= 4.0` | 48,580 |
| Positive users | 609 |
| Positive items | 6,298 |
| Train interactions | 38,833 |
| Validation interactions | 4,872 |
| Test interactions | 4,875 |
| Tỷ lệ positive | 48.18% |

### 3.2. Letterboxd

Letterboxd là dữ liệu được crawler riêng cho đồ án. Dữ liệu ban đầu có dạng interaction từ người dùng Letterboxd, sau đó được chuyển sang schema tương thích MovieLens.

**Nguồn raw:** `data/letterboxd/data/raw`

**Nguồn processed:** `data/processed/letterboxd`

Các file chính:

- `interactions.csv`, `interactions_cf.csv`: tương tác user-movie.
- `movies_seed.csv`, `movies_cf.csv`: danh sách phim.
- `users.csv`: thông tin user.
- `ratings.csv`: bản đã chuẩn hóa sang format MovieLens.
- `movies.csv`: phim đã chuẩn hóa.
- `letterboxd_user_mapping.csv`: mapping user Letterboxd sang `userId`.
- `letterboxd_movie_mapping.csv`: mapping movie Letterboxd sang `movieId`.
- `movie_catalog_enriched.parquet`: catalog sau khi enrich TMDb.

Thông tin hiện tại của Letterboxd processed:

| Thuộc tính | Giá trị |
|---|---:|
| Số interactions gốc | 503,761 |
| Số users gốc | 9,197 |
| Số movies gốc | 7,848 |
| Positive interactions với `rating >= 4.0` | 202,354 |
| Positive users | 8,985 |
| Positive items | 7,211 |
| Train interactions | 160,843 |
| Validation interactions | 20,669 |
| Test interactions | 20,842 |
| Tỷ lệ positive | 40.17% |

**Lưu ý về timestamp:** Letterboxd không có timestamp hành vi đáng tin cậy. Trường `created_at` là thời điểm crawler ghi dữ liệu, không phải thời điểm người dùng xem phim. Vì vậy, pipeline không dùng `created_at` để đánh giá temporal split. Dự án dùng synthetic random-per-user split ổn định theo seed.

### 3.3. TMDb Metadata

TMDb được dùng để làm giàu thông tin phim. MovieLens dùng `links.csv` để mapping sang `tmdbId`; Letterboxd dùng tìm kiếm theo `title + year`.

Các trường metadata chính:

- `tmdb_id`
- `overview`
- `poster_url`
- `release_date`
- `release_year`
- `tmdb_genres`
- `keywords`
- `popularity`
- `vote_average`
- `vote_count`
- `runtime_minutes`
- `original_language`
- `production_countries`
- `production_companies`
- `collection`
- `director`
- `writers`
- `cast`

Tình trạng enrich hiện tại:

| Catalog | Số phim | Có `tmdb_id` | Có poster | Có overview | Có director | Có cast |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens enriched | 9,742 | 9,621 | 9,617 | 9,620 | 9,617 | 9,591 |
| Letterboxd enriched | 7,848 | 7,337 | 7,321 | 7,328 | 7,318 | 7,291 |

## 4. Biểu diễn dữ liệu

### 4.1. Interaction matrix

Sau khi lọc `rating >= 4.0`, mỗi dòng tương tác được xem là positive feedback.

Mỗi interaction gồm:

```text
userId, movieId, rating, timestamp
```

Sau đó được encode thành:

```text
user_idx, item_idx, rating, timestamp
```

Trong đó:

- `userId`, `movieId`: ID gốc.
- `user_idx`, `item_idx`: ID liên tục từ 0, dùng để train model.
- `rating`: điểm explicit hoặc implicit score đã chuẩn hóa.
- `timestamp`: thời gian thật với MovieLens, synthetic timestamp với Letterboxd.

Interaction matrix được biểu diễn bằng ma trận sparse CSR:

```text
R ∈ R^(num_users x num_items)
R[u, i] = 1 nếu user u có positive interaction với item i
```

### 4.2. Positive item và negative item

Trong dự án:

- **Positive item:** phim mà user có `rating >= min_rating`, mặc định `min_rating = 4.0`.
- **Negative item:** phim mà user chưa có positive interaction trong tập đang dùng để train.

Negative item là assumed negative, tức không chắc user ghét phim đó, chỉ biết user chưa tương tác tích cực với phim đó.

Trong BPR training, mỗi sample có dạng:

```text
(user, positive_item, negative_item)
```

Mục tiêu:

```text
score(user, positive_item) > score(user, negative_item)
```

### 4.3. Content embedding

Với mỗi phim, hệ thống nối các trường metadata thành một văn bản:

```text
title + genres + overview + tmdb_genres + keywords + director + cast + ...
```

Sau đó mã hóa bằng một trong hai backend:

- `sbert`: dùng `sentence-transformers/all-mpnet-base-v2`.
- `tfidf`: dùng TF-IDF + TruncatedSVD để smoke test CPU nhanh.

Kết quả là vector:

```text
item_embedding[i] ∈ R^d
```

User content profile được tính bằng trung bình embedding các phim user đã thích:

```text
user_profile[u] = mean(item_embedding[i] for i in history[u])
```

## 5. Phương pháp học máy

### 5.1. Baseline models

Đồ án có nhiều baseline để so sánh:

| Model | Ý nghĩa |
|---|---|
| Random | Gợi ý ngẫu nhiên, làm mốc thấp nhất |
| Popularity | Gợi ý phim phổ biến nhất |
| ItemKNN Cosine | Dựa trên độ giống nhau giữa các item |
| UserKNN Cosine | Dựa trên độ giống nhau giữa các user |
| SVD Ranking | Matrix factorization bằng TruncatedSVD |
| BPR-MF | Matrix factorization tối ưu BPR loss |
| EASE | Linear recommender closed-form cho implicit feedback |
| SLIM ElasticNet | Sparse linear item-item model |
| implicit ALS | Alternating Least Squares, optional dependency |
| LightFM WARP | Learning-to-rank hybrid framework, optional dependency |
| NeuMF | Neural matrix factorization |

Các baseline này giúp chứng minh mô hình chính có được so sánh công bằng, thay vì chỉ báo cáo một kết quả đơn lẻ.

### 5.2. LightGCN

LightGCN là mô hình collaborative filtering trên đồ thị hai phía user-item.

Đồ thị gồm:

- Node user.
- Node item.
- Edge nếu user có positive interaction với item.

LightGCN không dùng feature transformation phức tạp như GCN truyền thống. Mô hình lan truyền embedding qua graph:

```text
E^(k+1) = A_norm E^(k)
```

Embedding cuối cùng là trung bình embedding qua các layer:

```text
E_final = mean(E^(0), E^(1), ..., E^(K))
```

Điểm dự đoán:

```text
score(u, i) = dot(user_embedding[u], item_embedding[i])
```

Loss sử dụng BPR:

```text
L = -log sigmoid(score(u, i_pos) - score(u, i_neg))
```

Lý do chọn LightGCN:

- Phù hợp với implicit feedback.
- Khai thác tốt cấu trúc user-item sparse.
- Nhẹ hơn các GCN phức tạp.
- Là baseline mạnh trong recommender system hiện đại.

### 5.3. Content-based model

Content-based model dùng metadata phim để xử lý trường hợp:

- User mới có ít lịch sử.
- Phim mới chưa có nhiều interaction.
- Cần giải thích vì sao phim được gợi ý.

Điểm content:

```text
score_content(u, i) = cosine(user_profile[u], item_embedding[i])
```

Với session-based recommendation, nếu người dùng chọn vài phim trong phiên hiện tại, hệ thống lấy trung bình embedding của các phim đó làm session profile.

### 5.4. Learned Two-Tower

Dự án có phiên bản learned Two-Tower thật:

- User tower: learned user embedding.
- Item tower: MLP nhận input là SBERT/TF-IDF metadata embedding.

Item tower:

```text
item_vector = MLP(content_embedding)
```

User tower:

```text
user_vector = Embedding(user_idx)
```

Score:

```text
score(u, i) = dot(user_vector[u], item_vector[i])
```

Loss cũng là BPR loss với triplet `(user, positive_item, negative_item)`.

### 5.5. Hybrid PDF-clean

Đây là mô hình chính có thể dùng để trình bày trong báo cáo vì không dùng feature từ các baseline mạnh khác.

Các thành phần:

- LightGCN score.
- Learned Two-Tower score.
- Content similarity score.
- Popularity score.

Mỗi score được chuẩn hóa Min-Max theo từng user:

```text
s_norm = (s - min(s)) / (max(s) - min(s))
```

Score cuối:

```text
score_hybrid =
    w_cf * score_lightgcn
  + w_two_tower * score_two_tower
  + w_content * score_content
  + w_popularity * score_popularity
```

Trọng số được tune trên validation bằng grid search. Metric chọn trọng số là `NDCG@10`. Popularity weight có thể bằng 0 nếu validation cho thấy popularity không giúp cải thiện.

### 5.6. Hybrid Strong Ranker

Ngoài mô hình chính, dự án có thêm phiên bản mạnh hơn để tối ưu kết quả:

- Candidate/component scores từ LightGCN, Two-Tower, EASE, ItemKNN, UserKNN, SVD, LightFM, ALS, popularity.
- Metadata feature: TMDb vote average, vote count, popularity, release year, runtime.
- User feature: độ dài lịch sử user.
- Genre overlap giữa lịch sử user và phim ứng viên.

Ranker ưu tiên dùng LightGBM LambdaRank; nếu thiếu dependency thì fallback sang `SGDClassifier(loss="log_loss")`.

Mô hình này không nên dùng làm phương pháp chính nếu mục tiêu báo cáo là chứng minh hybrid clean vượt baseline, vì nó dùng score của baseline làm feature. Tuy nhiên, nó hữu ích cho phần mở rộng và tối ưu kết quả thực tế.

## 6. Quy trình huấn luyện và inference

### 6.1. Pipeline dữ liệu

```text
Raw MovieLens / Letterboxd
  -> read data
  -> filter rating >= 4.0
  -> encode userId/movieId
  -> train/val/test split
  -> enrich TMDb metadata
  -> build sparse matrix
  -> build content embeddings
```

### 6.2. Pipeline train artifact chính

```text
ratings + catalog
  -> train LightGCN
  -> encode content embeddings
  -> train learned Two-Tower
  -> build user profiles
  -> tune hybrid weights on validation
  -> evaluate on test
  -> export artifacts
```

### 6.3. Pipeline inference

```text
FastAPI startup
  -> load artifacts
  -> receive request
  -> compute LightGCN/content/Two-Tower/popularity scores
  -> mask seen items
  -> top-K ranking
  -> return JSON response
```

API không train lại khi nhận request.

## 7. Cài đặt hệ thống

### 7.1. Công nghệ sử dụng

| Nhóm | Công cụ |
|---|---|
| Ngôn ngữ | Python |
| ML/Data | NumPy, Pandas, SciPy, scikit-learn |
| Deep Learning | PyTorch |
| Text embedding | sentence-transformers |
| Optional models | implicit, LightFM, LightGBM |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Visualization | Plotly |
| Storage | Parquet, JSON, NPY, CSV |
| Deployment local | Docker Compose |
| Training environment | Local CPU smoke test, Kaggle GPU |

Nếu sử dụng lại mã nguồn/gói phần mềm, báo cáo cần nêu rõ:

- `scikit-learn`: TF-IDF, TruncatedSVD, KMeans, GMM, SGDClassifier.
- `PyTorch`: LightGCN, BPR-MF, learned Two-Tower, NeuMF.
- `sentence-transformers`: SBERT metadata embedding.
- `FastAPI`: REST API.
- `Streamlit`: giao diện demo.
- `TMDb API`: metadata phim.
- `MovieLens`: dataset công khai.
- `Letterboxd`: dataset crawler cho đồ án.
- `LightGBM`, `implicit`, `LightFM`: optional comparison models.

## 8. Kết quả thí nghiệm

### 8.1. Metric đánh giá

Các metric chính:

- **Precision@K:** tỷ lệ phim đúng trong Top-K.
- **Recall@K:** tỷ lệ phim đúng được tìm thấy trong Top-K.
- **NDCG@K:** đánh giá cả độ đúng và vị trí xếp hạng; hit ở vị trí cao được thưởng nhiều hơn.
- **MRR:** thứ hạng nghịch đảo trung bình của hit đầu tiên.

Trong recommender system, đặc biệt với implicit feedback, ranking metrics quan trọng hơn RMSE. RMSE chỉ được dùng như baseline phụ cho SVD explicit rating.

### 8.2. Kết quả train artifact hiện tại

Kết quả với MovieLens artifact hiện tại:

| Split | Precision@10 | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Validation | 0.0345 | 0.0809 | 0.0629 | 0.0986 |
| Test | 0.0224 | 0.0476 | 0.0391 | 0.0644 |

Thông tin LightGCN:

- Đã bật LightGCN.
- BPR loss giảm từ khoảng 0.6902 xuống 0.1828 sau 50 epoch, cho thấy mô hình học được quan hệ user-item.
- Trọng số hybrid hiện tại: `cf_weight = 0.5`, `content_weight = 0.5`, `popularity_weight = 0.0`.

Kết quả với Letterboxd artifact hiện tại:

| Split | Precision@10 | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| Validation | 0.0383 | 0.1589 | 0.1196 | 0.1597 |
| Test | 0.0372 | 0.1505 | 0.1154 | 0.1545 |

Nhận xét:

- Letterboxd có kết quả NDCG@10 cao hơn MovieLens artifact hiện tại.
- Điều này phù hợp vì Letterboxd processed có nhiều interaction hơn và density trên tập positive sau xử lý tốt hơn.
- Popularity weight được tune về 0 trong artifact hiện tại, cho thấy trên validation, tín hiệu collaborative và content hữu ích hơn popularity thuần.

### 8.3. Kết quả comparison trên MovieLens

Kết quả comparison hiện có trong `reports/comparison_movielens`:

| Model | Precision@10 | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|---:|
| EASE | 0.0299 | 0.0612 | 0.0506 | 0.0898 |
| SVD Ranking | 0.0250 | 0.0604 | 0.0454 | 0.0727 |
| ItemKNN Cosine | 0.0275 | 0.0554 | 0.0438 | 0.0736 |
| UserKNN Cosine | 0.0255 | 0.0554 | 0.0435 | 0.0736 |
| Hybrid Weighted No Popularity | 0.0235 | 0.0546 | 0.0426 | 0.0698 |
| Hybrid Weighted Full | 0.0201 | 0.0432 | 0.0377 | 0.0688 |
| Popularity Only | 0.0194 | 0.0397 | 0.0365 | 0.0677 |
| LightGCN Only | 0.0184 | 0.0390 | 0.0359 | 0.0667 |
| TF-IDF Only | 0.0063 | 0.0171 | 0.0113 | 0.0184 |
| Random | 0.0013 | 0.0018 | 0.0030 | 0.0084 |

Nhận xét:

- EASE đang là baseline mạnh nhất trong comparison MovieLens hiện tại.
- Hybrid weighted chưa vượt EASE trên MovieLens local CPU run, cho thấy cần tiếp tục train/tune trên Kaggle bằng SBERT và epochs lớn hơn.
- Content-only thấp hơn rõ rệt, chứng minh chỉ dùng metadata không đủ cho cá nhân hóa.
- Random gần như bằng 0, xác nhận evaluation pipeline hợp lý.
- BPR-MF trong run hiện tại thấp, có thể do số epoch/siêu tham số chưa phù hợp.

### 8.4. Thí nghiệm cần chạy cho bản nộp cuối

Để bản báo cáo cuối mạnh hơn, cần chạy trên Kaggle:

1. `letterboxd-pdf-clean`: LightGCN + SBERT + learned Two-Tower + weighted hybrid.
2. `letterboxd-strong`: full comparison với EASE, ItemKNN, UserKNN, SVD, ALS, LightFM, strong ranker.
3. Export `comparison_results.csv/json/md`.
4. Cập nhật bảng kết quả trong mục 8.2 và 8.3.

## 9. Các chức năng chính của hệ thống

### 9.1. FastAPI backend

Các endpoint chính:

| Endpoint | Chức năng |
|---|---|
| `GET /health` | Kiểm tra trạng thái service và artifact |
| `GET /movies` | Tìm kiếm phim theo query |
| `POST /recommendations` | Sinh gợi ý Top-K |
| `POST /recommend` | Alias tương thích |
| `GET /users` | Danh sách user có trong artifact |
| `GET /users/{user_id}/history` | Lịch sử user |
| `GET /movies/trending` | Phim thịnh hành |
| `GET /movies/top-rated` | Phim đánh giá cao |
| `GET /movies/latest` | Phim mới |
| `GET /movies/genre/{genre}` | Phim theo thể loại |
| `GET /movies/{movie_id}` | Chi tiết phim |
| `GET /movies/{movie_id}/similar` | Phim tương tự |
| `GET /model-info` | Thông tin mô hình và metric |
| `POST /rate` | Lưu rating sidecar |
| `GET /rate/{user_id}/{movie_id}` | Lấy rating mới nhất của user |
| `POST /chat` | Chatbot tư vấn phim |

### 9.2. Streamlit UI

Giao diện chính gồm:

- Trang chính: gợi ý cá nhân hóa, phim mới, phim thịnh hành, phim đánh giá cao, phim theo thể loại.
- Search: tìm phim theo tên.
- Detail: xem thông tin phim, poster, overview, director, cast.
- Similar movies: phim tương tự theo content embedding.
- Rating: user có thể chấm phim, lưu vào sidecar CSV.
- History: xem lịch sử user.
- Metrics: xem kết quả đánh giá.
- Chatbot: hỏi gợi ý phim bằng tiếng Việt.

### 9.3. EDA Dashboard

EDA dashboard gồm:

- Thống kê số user, số item, số interaction, sparsity.
- Phân phối rating.
- Phân phối metadata: release year, runtime, popularity, vote average.
- Top thể loại.
- Top phim được tương tác nhiều.
- User segmentation bằng KMeans/GMM.

### 9.4. Embedding Visualization

Script visualization giảm chiều embedding phim bằng PCA hoặc t-SNE và sinh HTML tương tác. Mục đích:

- Minh họa không gian embedding nội dung.
- Kiểm tra phim cùng thể loại/cùng nội dung có xu hướng gần nhau không.
- Phục vụ phần trình bày trực quan trong báo cáo.

## 10. Cấu trúc mã nguồn

```text
src/recommender/
  config.py
  data/
    movielens.py
    letterboxd.py
    tmdb.py
  models/
    lightgcn.py
    losses.py
    two_tower.py
    learned_two_tower.py
    baselines.py
    matrix_factorization.py
    rankers.py
    svd.py
  eval/
    metrics.py
  experiments/
    comparison.py
  inference/
    artifacts.py
    recommender.py
    ratings_store.py
  rag/
    retriever.py
    chatbot.py
  analysis/
    eda.py

api/
  main.py

app/
  streamlit_app.py
  eda_app.py

scripts/
  download_movielens.py
  enrich_tmdb.py
  prepare_letterboxd.py
  train.py
  compare_models.py
  train_strong_hybrid.py
  visualize_embeddings.py
```

### 10.1. Các class/function quan trọng

| File | Thành phần | Vai trò |
|---|---|---|
| `data/movielens.py` | `read_movielens` | Đọc MovieLens CSV/DAT |
| `data/movielens.py` | `prepare_interactions` | Lọc positive, encode ID, split train/val/test |
| `data/letterboxd.py` | `materialize_letterboxd` | Convert Letterboxd sang format MovieLens-compatible |
| `data/letterboxd.py` | `enrich_letterboxd_catalog` | Enrich Letterboxd bằng TMDb search |
| `data/tmdb.py` | `TMDBClient` | Gọi TMDb API có retry/cache |
| `models/losses.py` | `sample_bpr_triplets` | Sinh triplet BPR |
| `models/losses.py` | `bpr_loss` | Tính BPR loss |
| `models/lightgcn.py` | `LightGCNModel` | Mô hình LightGCN |
| `models/two_tower.py` | `encode_item_texts` | Tạo SBERT/TF-IDF item embeddings |
| `models/learned_two_tower.py` | `LearnedTwoTowerRecommender` | Learned user tower + item MLP |
| `models/baselines.py` | `EASERecommender`, `ItemKNNRecommender` | Baseline comparison |
| `models/rankers.py` | `WeightedHybridRecommender` | Tune weighted hybrid |
| `models/rankers.py` | `StrongHybridRankerRecommender` | Learned ranker mạnh |
| `eval/metrics.py` | `evaluate_score_fn` | Evaluate ranking metrics có mask train items |
| `experiments/comparison.py` | `run_comparison` | Orchestrate comparison suite |
| `inference/artifacts.py` | `save_artifact_bundle`, `load_artifact_bundle` | Lưu/đọc artifact |
| `inference/recommender.py` | `HybridArtifactRecommender` | Logic inference chính |
| `rag/chatbot.py` | `MovieRAGChatbot` | Chatbot tư vấn phim |
| `analysis/eda.py` | `user_segmentation` | Phân cụm user cho EDA |

## 11. Hướng dẫn chạy chương trình

### 11.1. Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Điền `TMDB_API_KEY` nếu cần enrich TMDb. Điền `OPENAI_API_KEY` nếu muốn chatbot gọi LLM; nếu không điền, chatbot vẫn chạy chế độ local.

### 11.2. Train nhanh local bằng TF-IDF

```bash
PYTHONPATH=src:. python scripts/train.py \
  --raw-dir data/processed/letterboxd \
  --enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
  --artifacts-dir artifacts/letterboxd \
  --content-backend tfidf \
  --train-lightgcn \
  --epochs 10 \
  --min-rating 4.0
```

### 11.3. Train đầy đủ trên Kaggle

```bash
PYTHONPATH=src:. python scripts/train.py \
  --raw-dir data/processed/letterboxd \
  --enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
  --artifacts-dir artifacts/letterboxd_pdf_clean \
  --content-backend sbert \
  --sbert-model sentence-transformers/all-mpnet-base-v2 \
  --train-lightgcn \
  --train-two-tower \
  --lightgcn-dim 128 \
  --lightgcn-layers 3 \
  --epochs 100 \
  --batch-size 8192 \
  --device cuda \
  --hybrid-grid-step 0.05 \
  --min-rating 4.0
```

### 11.4. So sánh mô hình

```bash
PYTHONPATH=src:. python scripts/compare_models.py \
  --dataset letterboxd \
  --letterboxd-dir data/processed/letterboxd \
  --letterboxd-enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
  --content-backend sbert \
  --preset letterboxd-pdf-clean \
  --k 10 \
  --device cuda \
  --output-dir reports/comparison_letterboxd_pdf_clean
```

```bash
PYTHONPATH=src:. python scripts/compare_models.py \
  --dataset letterboxd \
  --letterboxd-dir data/processed/letterboxd \
  --letterboxd-enriched-catalog data/processed/letterboxd/movie_catalog_enriched.parquet \
  --content-backend sbert \
  --preset letterboxd-strong \
  --models full \
  --k 10 \
  --epochs 100 \
  --mf-dim 128 \
  --batch-size 8192 \
  --device cuda \
  --max-ease-items 5000 \
  --max-slim-items 3000 \
  --max-ranker-samples 500000 \
  --output-dir reports/comparison_letterboxd_strong
```

### 11.5. Chạy API/UI

```bash
PYTHONPATH=src:. uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
```

### 11.6. Chạy EDA và visualization

```bash
PYTHONPATH=src:. streamlit run app/eda_app.py
```

```bash
PYTHONPATH=src:. python scripts/visualize_embeddings.py \
  --artifacts-dir artifacts/letterboxd_pdf_clean \
  --output-dir reports/embedding_visualization_letterboxd \
  --method tsne \
  --sample-size 2500 \
  --top-genres 8
```

### 11.7. Chạy test

```bash
PYTHONPATH=src:. pytest
```

## 12. Khó khăn và cách giải quyết

### 12.1. TMDb API bị reset/refused

Khi enrich nhiều phim, local có thể gặp lỗi `connection reset by peer` hoặc `connection refused`.

Giải pháp:

- Thêm cache JSON để resume.
- Thêm retry, timeout, sleep giữa các request.
- Cho phép chạy riêng enrich trên Kaggle/cloud.
- Không commit API key.

### 12.2. Letterboxd thiếu timestamp hành vi

Letterboxd crawler có `created_at`, nhưng đây là thời điểm crawl, không phải thời điểm user xem phim.

Giải pháp:

- Không dùng `created_at` làm thời gian thật.
- Dùng synthetic random-per-user split ổn định theo seed.
- Ghi rõ trong báo cáo để tránh hiểu nhầm temporal evaluation.

### 12.3. Dữ liệu implicit không có negative thật

Trong recommender implicit feedback, không xem phim không đồng nghĩa với ghét phim.

Giải pháp:

- Dùng negative sampling từ unobserved items.
- Dùng BPR loss thay vì regression loss.
- Đánh giá bằng ranking metrics.

### 12.4. Cold-start

Collaborative filtering yếu với phim mới hoặc user mới.

Giải pháp:

- Dùng metadata TMDb và SBERT/TF-IDF embeddings.
- Hỗ trợ session context.
- Tạo content similarity và similar movies.

### 12.5. Chi phí train SBERT/LightGCN

Local CPU khó chạy full training.

Giải pháp:

- Local dùng TF-IDF smoke test.
- Kaggle GPU dùng SBERT, LightGCN epochs lớn hơn.
- Cache content embeddings để tránh encode lại nhiều lần.

### 12.6. So sánh mô hình công bằng

Nếu hybrid dùng feature từ baseline rồi so sánh với baseline, kết luận có thể không công bằng.

Giải pháp:

- Tách `hybrid_pdf_clean` để trình bày phương pháp chính.
- Tách `hybrid_strong_ranker` để tối ưu thực tế.
- Báo cáo rõ hai nhóm mô hình.

## 13. Khám phá và kết luận

Một số kết luận từ quá trình thực hiện:

1. Ranking metrics phù hợp hơn RMSE cho bài toán gợi ý phim implicit feedback.
2. Content-only không đủ mạnh để cá nhân hóa, nhưng rất hữu ích cho cold-start và giải thích.
3. Collaborative filtering khai thác tốt hành vi user-item nhưng phụ thuộc vào độ dày interaction.
4. Hybrid cần tune trọng số trên validation; popularity không phải lúc nào cũng giúp.
5. EASE là baseline rất mạnh trên MovieLens small, cần đưa vào comparison để báo cáo thuyết phục.
6. Letterboxd sau xử lý có lượng interaction lớn, là dataset phù hợp để tối ưu kết quả chính.
7. Metadata TMDb làm tăng giá trị demo vì có poster, overview, director, cast và hỗ trợ chatbot/visualization.

Kết luận chung: đồ án đã xây dựng được một hệ thống gợi ý phim hoàn chỉnh từ data pipeline, model training, model comparison, artifact export, API inference, Streamlit UI, EDA dashboard và RAG chatbot. Hệ thống có khả năng mở rộng sang các mô hình mạnh hơn và có quy trình đánh giá rõ ràng.

## 14. Phân công công việc nhóm

Nếu nhóm có 5 thành viên, có thể phân công như sau:

| Thành viên | Công việc |
|---|---|
| Thành viên 1 | Data pipeline MovieLens, Letterboxd, xử lý split và mapping |
| Thành viên 2 | TMDb enrichment, cache, xử lý lỗi API, catalog metadata |
| Thành viên 3 | LightGCN, BPR loss, negative sampling, train artifact |
| Thành viên 4 | Content model, learned Two-Tower, hybrid scorer, comparison suite |
| Thành viên 5 | FastAPI, Streamlit UI, EDA dashboard, chatbot, test và demo |

Tất cả thành viên cần tham gia trình bày:

- Người 1: giới thiệu bài toán và dữ liệu.
- Người 2: trình bày pipeline xử lý/enrich.
- Người 3: trình bày LightGCN và BPR.
- Người 4: trình bày hybrid, comparison và kết quả.
- Người 5: demo API/UI/chatbot/EDA và kết luận.

## 15. Dàn ý trình bày 15 phút

Gợi ý phân bổ thời gian:

| Thời lượng | Nội dung |
|---:|---|
| 1 phút | Giới thiệu bài toán quá tải lựa chọn phim |
| 2 phút | Dữ liệu MovieLens, Letterboxd, TMDb |
| 3 phút | Phương pháp: LightGCN, content embedding, Two-Tower, hybrid |
| 2 phút | Baseline và metric đánh giá |
| 2 phút | Kết quả thí nghiệm |
| 3 phút | Demo API/UI: gợi ý, tìm kiếm, chi tiết phim, chatbot |
| 1 phút | Khó khăn và cách giải quyết |
| 1 phút | Kết luận và hướng phát triển |

## 16. Hướng phát triển

Một số hướng có thể phát triển tiếp:

- Train đầy đủ SBERT + LightGCN + learned Two-Tower trên Kaggle GPU.
- Tối ưu hyperparameter cho LightGCN, EASE, BPR-MF.
- Dùng LambdaRank/LightGBM cho production ranker.
- Thêm online learning từ rating sidecar.
- Thêm diversity/re-ranking để tránh gợi ý quá nhiều phim giống nhau.
- Thêm explainability chi tiết hơn: cùng đạo diễn, cùng diễn viên, cùng keyword.
- Deploy public bằng Docker hoặc cloud.

