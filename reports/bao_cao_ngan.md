# Báo cáo ngắn: Hệ thống gợi ý phim hybrid

## 1. Bài toán

Hệ thống giải quyết bài toán quá tải lựa chọn phim bằng cách tạo danh sách gợi ý cá nhân hóa cho từng người dùng hoặc phiên xem mới. Hai thách thức chính là dữ liệu rating thưa và cold-start cho người dùng/phim mới.

## 2. Dữ liệu

- MovieLens: `ratings.csv`, `movies.csv`, `links.csv`.
- TMDb: poster, overview, thể loại mở rộng, keyword, đạo diễn, biên kịch, diễn viên, collection/franchise, quốc gia sản xuất, studio, runtime, popularity và vote statistics.
- Letterboxd: dữ liệu crawler gồm users, movies và interactions. Bản CF-ready được chuyển sang format tương thích MovieLens; do `created_at` là thời điểm crawl, không phải thời điểm xem, đánh giá Letterboxd dùng split random ổn định theo user thay vì temporal split.
- Dữ liệu sau xử lý được lưu dạng `Parquet`; artifacts mô hình lưu trong thư mục `artifacts/`.

## 3. Kiến trúc phương pháp

- Baseline SVD: tạo mốc so sánh RMSE và ranking.
- LightGCN: học embedding user/item trên đồ thị hai phía user-phim, tối ưu bằng BPR loss.
- SBERT Two-Tower: mã hóa metadata phim thành vector nội dung; user profile là trung bình vector các phim đã tương tác.
- Hybrid scorer: chuẩn hóa Min-Max điểm collaborative, content và popularity, sau đó dò trọng số tốt nhất trên validation theo NDCG@K.

## 4. Hệ thống phần mềm

- FastAPI phục vụ `/health`, `/movies`, `/recommendations`.
- Streamlit cung cấp giao diện tiếng Việt để chọn user hoặc session context và xem poster/lý do gợi ý.
- Docker Compose chạy hai service API và UI, mount chung `data/` và `artifacts/`.

## 5. Đánh giá

Các metric chính:
- Precision@K: tỷ lệ phim gợi ý đúng trong top K.
- Recall@K: tỷ lệ phim đúng được tìm thấy.
- NDCG@K: đánh giá cả độ đúng và thứ tự xếp hạng.
- MRR: thứ hạng nghịch đảo trung bình của hit đầu tiên.
- RMSE: dùng cho baseline SVD.

Kết quả thực nghiệm được ghi vào `artifacts/metrics.json` sau mỗi lần train.

## 6. Phân công nhóm 5 thành viên

- Data Engineer: tải MovieLens, enrich TMDb, cache và làm sạch dữ liệu.
- ML Engineer 1: LightGCN, BPR loss, negative sampling.
- ML Engineer 2: SBERT Two-Tower, hybrid scorer, tuning trọng số.
- Backend Engineer: FastAPI, artifact loading, contract JSON.
- Frontend/QA Engineer: Streamlit UI, metrics view, tests và demo.

## 7. Cách demo

1. Tải MovieLens và enrich TMDb.
2. Train trên Kaggle hoặc local smoke mode.
3. Copy artifacts về repo.
4. Chạy FastAPI và Streamlit.
5. Trình bày hai luồng: user đã có lịch sử và cold-start bằng session context.
