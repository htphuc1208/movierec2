# Phân tích Khám phá Dữ liệu (EDA) — Letterboxd (crawled)

## 1. Tổng quan dữ liệu (Data Overview)

| Thuộc tính | Giá trị |
|---|---:|
| Số người dùng | 9,197 |
| Số phim có tương tác | 7,848 |
| Tổng interactions | 503,761 |
| Sparsity | 99.30% |
| Rating trung bình | 3.41 |
| Rating trung vị | 3.5 |
| Rating min | 0.5 |
| Rating max | 5.0 |

### 1.1. Chất lượng dữ liệu

**Missing values trong bảng interactions:**

- `userId`: 0 (0.00%)
- `movieId`: 0 (0.00%)
- `rating`: 0 (0.00%)
- `timestamp`: 0 (0.00%)

**Missing values trong catalog (các cột quan trọng):**

- `title`: 0 (0.0%)
- `genres`: 0 (0.0%)
- `tmdb_genres`: 0 (0.0%)
- `overview`: 0 (0.0%)
- `release_year`: 0 (0.0%)
- `popularity`: 0 (0.0%)
- `vote_average`: 0 (0.0%)
- `director`: 0 (0.0%)

## 2. Phân phối Rating

![Phân phối rating](/home/phucht/movierec3/reports/eda_letterboxd/01_rating_distribution.png)

- Tỷ lệ positive (rating ≥ 4.0): **202,354** / 503,761 = **40.2%**
- Rating trung bình: **3.41** — cho thấy xu hướng rating trung bình

## 3. Phân tích hoạt động User

| Thuộc tính | Giá trị |
|---|---:|
| Trung bình rating/user | 54.8 |
| Trung vị rating/user | 60.0 |
| Min rating/user | 10 |
| Max rating/user | 154 |
| Users có ≤ 20 ratings (cold-start) | 1136 (12.4%) |
| Users có > 100 ratings (power users) | 280 (3.0%) |

![User activity](/home/phucht/movierec3/reports/eda_letterboxd/02_user_activity.png)

**Nhận xét:** Phân phối hoạt động user theo dạng **power-law** — một số ít user rất tích cực (power users), đa số user có ít ratings. Đây là đặc trưng phổ biến của recommendation datasets.

## 4. Phân tích Long-tail phim

| Thuộc tính | Giá trị |
|---|---:|
| Trung bình ratings/phim | 64.2 |
| Trung vị ratings/phim | 13.0 |
| Phim chỉ có 1 rating | 0 (0.0%) |
| Phim có ≤ 5 ratings | 934 (11.9%) |
| Top-1% phim chiếm bao nhiêu % interactions | 23.0% |

![Long-tail analysis](/home/phucht/movierec3/reports/eda_letterboxd/03_longtail_items.png)

**Nhận xét:** Hiện tượng **long-tail** rõ rệt — 17% phim phổ biến nhất chiếm 80% tổng interactions. Phim ở long-tail (ít interaction) khó recommend bằng CF thuần, cần content-based để bổ trợ.

## 5. Phân tích theo thời gian

![Temporal patterns](/home/phucht/movierec3/reports/eda_letterboxd/04_temporal_patterns.png)

## 6. Phân tích thể loại và nội dung

![Genre analysis](/home/phucht/movierec3/reports/eda_letterboxd/05_genre_analysis.png)

![Release year](/home/phucht/movierec3/reports/eda_letterboxd/06_release_year.png)

## 7. Phân tích Sparsity

- Ma trận user-item có kích thước: **9,197 × 7,848** = **72,178,056** ô
- Chỉ có **503,761** ô có giá trị → Sparsity = **99.30%**
- Đây là mức sparsity rất cao, đặc trưng cho recommendation datasets.

![Sparsity matrix](/home/phucht/movierec3/reports/eda_letterboxd/07_sparsity_matrix.png)

## 8. Phân cụm người dùng (User Segmentation)

Sử dụng K-Means clustering trên feature trung bình của các phim mà user đã xem (thể loại, popularity, vote_average, release_year).

![User segmentation](/home/phucht/movierec3/reports/eda_letterboxd/08_user_segmentation.png)

- **Cụm 0: fan Comedy**: Top thể loại: Comedy (0.41), Drama (0.36), Adventure (0.31)
- **Cụm 1: fan Action**: Top thể loại: Action (0.42), Adventure (0.42), Science Fiction (0.35)
- **Cụm 2: fan Drama**: Top thể loại: Drama (0.57), Thriller (0.28), Comedy (0.28)
- **Cụm 3: fan Drama**: Top thể loại: Drama (0.33), Horror (0.28), Comedy (0.27)

## 9. Top phim được tương tác nhiều nhất

![Top movies](/home/phucht/movierec3/reports/eda_letterboxd/09_top_movies.png)

## 10. Tổng kết EDA

### Các phát hiện chính:

1. **Sparsity cao (99.3%)** — cần kỹ thuật CF hiệu quả (LightGCN, EASE) và content-based để bổ trợ.
2. **Long-tail rõ rệt** — 17% phim chiếm 80% interactions. Content-based giúp recommend phim ít tương tác.
3. **Power-law user activity** — đa số user có ít ratings, cần xử lý cold-start bằng popularity fallback.
4. **Rating nghiêng về tích cực** — trung bình 3.41, phù hợp dùng implicit feedback (threshold ≥ 4.0).
5. **User segmentation** — phân được 4 nhóm user rõ ràng theo sở thích thể loại.

### Ý nghĩa cho thiết kế hệ thống:

- Hybrid approach (CF + Content) phù hợp vì CF tốt cho warm users, Content giúp cold-start và long-tail items.
- LightGCN khai thác graph structure user-item, phù hợp với dữ liệu implicit feedback.
- Min-Max normalization cho hybrid scoring để cân bằng scale giữa các component.
- User segmentation giúp hiểu đối tượng và personalize recommendation strategy.
