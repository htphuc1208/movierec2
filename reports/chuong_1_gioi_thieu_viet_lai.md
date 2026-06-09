# CHƯƠNG 1. GIỚI THIỆU

## 1.1. Bối cảnh

Trong những năm gần đây, các nền tảng xem phim trực tuyến và cộng đồng đánh giá phim như Netflix, Disney+, HBO Max, IMDb, Letterboxd hay các dịch vụ streaming khác phát triển mạnh mẽ. Người dùng có thể tiếp cận một lượng phim rất lớn, thuộc nhiều thể loại, quốc gia, thời kỳ phát hành và phong cách khác nhau. Sự phong phú này giúp người dùng có nhiều lựa chọn hơn, nhưng đồng thời cũng tạo ra vấn đề quá tải thông tin: người dùng khó tìm được những bộ phim thật sự phù hợp với sở thích cá nhân nếu chỉ duyệt thủ công.

Hệ thống gợi ý phim được xây dựng để giải quyết vấn đề trên. Thay vì yêu cầu người dùng tự tìm kiếm trong hàng nghìn bộ phim, hệ thống phân tích lịch sử đánh giá, tương tác và thông tin nội dung phim để tạo danh sách phim có khả năng phù hợp nhất với từng người dùng. Trong miền phim ảnh, hai nhóm thông tin đặc biệt quan trọng là:

- Dữ liệu hành vi người dùng, ví dụ user đã đánh giá cao hoặc tương tác tích cực với phim nào.
- Metadata phim, ví dụ thể loại, mô tả nội dung, từ khóa, đạo diễn, diễn viên, năm phát hành, độ phổ biến và poster.

Tuy nhiên, chỉ sử dụng một loại tín hiệu thường không đủ. Mô hình Collaborative Filtering có thể học tốt từ hành vi người dùng, nhưng gặp khó khăn với user mới, user ít lịch sử hoặc phim ít tương tác. Content-based Filtering có thể khai thác metadata để hiểu nội dung phim, nhưng nếu dùng đơn lẻ thì mức độ cá nhân hóa thường chưa cao và thiếu đi tính bất ngờ. Vì vậy, đồ án xây dựng một hệ thống gợi ý phim hybrid, kết hợp dữ liệu tương tác và metadata để cải thiện chất lượng gợi ý trong nhiều tình huống sử dụng khác nhau.

## 1.2. Bài toán

Đề tài tập trung vào bài toán gợi ý phim cá nhân hóa dạng Top-K recommendation. Với một người dùng hoặc một ngữ cảnh phiên xem, hệ thống cần xếp hạng các phim chưa xem và trả về danh sách top-k những phim phù hợp nhất.

### Đầu vào

Dữ liệu đầu vào của hệ thống gồm:

- Thông tin định danh người dùng: `user_id`.
- Lịch sử đánh giá hoặc tương tác người dùng-phim.
- Thông tin phim từ MovieLens hoặc Letterboxd: `movieId`, `title`, `genres`.
- Metadata phim từ TMDb: `overview`, `tmdb_genres`, `keywords`, `director`, `cast`, `writers`, `poster_url`, `release_year`, `runtime_minutes`, `vote_average`, `vote_count`, `popularity`.
- Ngữ cảnh phiên xem nếu có: danh sách phim người dùng chọn trong phiên hiện tại (`session_context`).
- Số lượng phim cần gợi ý: `top_k`.
- Chế độ mô hình: `hybrid`, `lightgcn`, `two_tower`, `content`, `popularity`.

### Đầu ra

Kết quả đầu ra là danh sách Top-K phim được xếp hạng theo mức độ phù hợp. Mỗi phim trong danh sách gợi ý gồm các thông tin:

- `movie_id`
- `tmdb_id`
- `title`
- `score`
- `genres`
- `poster_url`
- `overview`
- `director`
- `cast`
- `explanation_tags`

### Phát biểu bài toán

Cho tập người dùng `U`, tập phim `I` và tập tương tác quan sát được `R`. Với mỗi người dùng `u ∈ U`, hệ thống cần học hàm tính điểm:

```text
score(u, i), với i ∈ I
```

Trong đó `score(u, i)` biểu diễn mức độ phù hợp của phim `i` đối với người dùng `u`. Sau đó, hệ thống loại bỏ các phim user đã xem trong tập train, sắp xếp các phim còn lại theo điểm giảm dần và chọn ra Top-K phim:

```text
TopK(u) = arg top K score(u, i)
```

Trong dự án này, tương tác tích cực được xác định bằng điều kiện:

```text
rating >= 4.0
```

Do đó, bài toán được xem như implicit feedback ranking: hệ thống không chỉ dự đoán rating tuyệt đối, mà tập trung vào việc xếp hạng phim nào nên được đề xuất trước.

## 1.3. Kịch bản ứng dụng

Hệ thống hỗ trợ hai kịch bản sử dụng chính.

### Kịch bản 1: Người dùng đã có lịch sử đánh giá

Người dùng đã có lịch sử tương tác trong MovieLens hoặc Letterboxd. Khi người dùng truy cập hệ thống, hệ thống dùng `user_id` để truy xuất lịch sử các phim đã đánh giá tích cực. Từ lịch sử này, hệ thống tính các nguồn điểm:

- Điểm collaborative từ LightGCN.
- Điểm từ Learned Two-Tower.
- Điểm content similarity từ metadata phim.
- Điểm popularity làm tín hiệu bổ trợ.

Các điểm này được chuẩn hóa và kết hợp bằng mô hình hybrid để tạo danh sách phim gợi ý cá nhân hóa.

### Kịch bản 2: Người dùng mới hoặc phiên xem mới

Trong trường hợp user chưa có lịch sử hoặc không chọn `user_id`, hệ thống vẫn có thể gợi ý dựa trên `session_context`. Người dùng chọn một vài phim yêu thích trong phiên hiện tại, hệ thống lấy trung bình content embedding của các phim này để tạo session profile. Sau đó, hệ thống tìm các phim có metadata tương tự với session profile và kết hợp thêm popularity nếu cần.

Kịch bản này giúp hệ thống xử lý cold-start tốt hơn so với collaborative filtering thuần.

### Kịch bản 3: Hỏi đáp bằng ngôn ngữ tự nhiên

Ngoài gợi ý theo user, hệ thống còn có chatbot tư vấn phim. Người dùng có thể hỏi:

```text
"Gợi ý phim khoa học viễn tưởng về không gian"
"Có phim nào giống The Dark Knight không?"
"Tôi muốn xem phim drama cảm động"
```

Chatbot truy xuất phim liên quan từ catalog dựa trên metadata và trả lời bằng tiếng Việt. Đây là phần mở rộng giúp người dùng tìm phim theo nhu cầu tự nhiên hơn.

## 1.4. Mục tiêu

Mục tiêu của đề tài là xây dựng và đánh giá một hệ thống gợi ý phim hybrid có khả năng cá nhân hóa danh sách phim cho người dùng, đồng thời có thể demo qua giao diện web và API.

Cụ thể, đề tài hướng tới các mục tiêu sau:

1. Xây dựng pipeline xử lý dữ liệu từ MovieLens và Letterboxd theo schema thống nhất.
2. Làm giàu catalog phim bằng metadata từ TMDb.
3. Chuyển dữ liệu rating thành implicit positive feedback với ngưỡng `rating >= 4.0`.
4. Biểu diễn dữ liệu dưới dạng user-item sparse matrix, graph user-item và content embedding.
5. Triển khai các baseline phổ biến như Popularity, UserKNN, ItemKNN, SVD Ranking, BPR-MF và EASE.
6. Triển khai mô hình collaborative filtering bằng LightGCN.
7. Triển khai mô hình content-based và Learned Two-Tower dựa trên metadata phim.
8. Xây dựng mô hình Hybrid kết hợp collaborative score, content score, two-tower score và popularity score.
9. Đánh giá mô hình bằng các ranking metrics như Precision@10, Recall@10, NDCG@10 và MRR.
10. So sánh kết quả trên hai tập dữ liệu MovieLens và Letterboxd.
11. Xây dựng hệ thống demo gồm FastAPI, Streamlit và chatbot tư vấn phim.

## 1.5. Phạm vi nghiên cứu

Trong khuôn khổ đồ án, phạm vi nghiên cứu được giới hạn như sau.

### Về dữ liệu

- Sử dụng MovieLens làm tập dữ liệu benchmark chuẩn.
- Sử dụng Letterboxd làm tập dữ liệu crawler thực tế để đánh giá thêm khả năng tổng quát hóa.
- Sử dụng TMDb để bổ sung metadata phim.
- Chỉ dùng các trường dữ liệu phục vụ bài toán gợi ý phim, không khai thác thông tin nhạy cảm của người dùng.
- Với Letterboxd, timestamp dùng trong split là synthetic random timestamp ổn định theo user vì thời gian crawl không phản ánh thời điểm xem phim thật.

### Về bài toán

- Tập trung vào Top-K movie recommendation.
- Ưu tiên ranking phim phù hợp hơn là dự đoán rating tuyệt đối.
- Không nghiên cứu các bài toán ngoài phạm vi như dự đoán doanh thu, phân loại cảm xúc review, nhận diện hình ảnh poster hoặc dự báo xu hướng thị trường phim.

### Về phương pháp

Các nhóm phương pháp trong phạm vi đồ án gồm:

- Popularity-based Recommendation.
- Collaborative Filtering.
- Matrix Factorization.
- Graph-based Recommendation với LightGCN.
- Content-based Recommendation từ metadata phim.
- Learned Two-Tower.
- Hybrid Recommendation.
- RAG-based movie chatbot ở mức demo.

### Về hệ thống

- Hệ thống phục vụ mục đích học tập, nghiên cứu và demo.
- Inference dùng artifact đã huấn luyện sẵn, không train lại khi nhận request.
- Demo được triển khai bằng FastAPI và Streamlit.
- Không đặt mục tiêu triển khai thương mại, chịu tải lớn hoặc real-time training ở quy mô production.

## 1.6. Đóng góp của đề tài

Đề tài có các đóng góp chính sau.

### Đóng góp về dữ liệu

- Chuẩn hóa dữ liệu MovieLens và Letterboxd về cùng format huấn luyện.
- Làm giàu thông tin phim bằng TMDb metadata gồm overview, genre mở rộng, keyword, đạo diễn, diễn viên, poster, năm phát hành và các chỉ số vote/popularity.
- Xây dựng artifact catalog phục vụ cả training, inference, UI và chatbot.

### Đóng góp về phương pháp

- Triển khai và so sánh nhiều nhóm mô hình gợi ý trên cùng protocol đánh giá.
- Xây dựng mô hình hybrid kết hợp LightGCN, Learned Two-Tower, content-based score và popularity score.
- Sử dụng metadata phim để hỗ trợ cold-start, session-based recommendation và long-tail items.
- Thực hiện ablation như so sánh hybrid có/không có TMDb metadata để đánh giá vai trò của metadata.

### Đóng góp về thực nghiệm

- Đánh giá mô hình trên cả MovieLens và Letterboxd.
- Sử dụng các ranking metrics phù hợp với recommender system: Precision@10, Recall@10, NDCG@10 và MRR.
- Phân tích riêng các nhóm sparse users và long-tail items để hiểu mô hình phù hợp trong từng trường hợp.

### Đóng góp về hệ thống

- Xây dựng pipeline hoàn chỉnh từ thu thập dữ liệu, làm sạch, feature engineering, huấn luyện, đánh giá đến export artifact.
- Xây dựng API inference để sinh gợi ý mà không cần train lại.
- Xây dựng giao diện Streamlit để demo tìm kiếm phim, xem chi tiết, xem phim tương tự, gợi ý theo user hoặc session.
- Tích hợp chatbot truy xuất metadata phim và trả lời tư vấn bằng tiếng Việt.

## 1.7. Cấu trúc báo cáo

Báo cáo được tổ chức như sau:

- Chương 1 giới thiệu bối cảnh, bài toán, mục tiêu, phạm vi và đóng góp của đề tài.
- Chương 2 trình bày cơ sở lý thuyết về recommender system, collaborative filtering, content-based filtering, graph recommendation và hybrid recommendation.
- Chương 3 mô tả dữ liệu sử dụng, quy trình thu thập, làm sạch, enrich metadata và phân tích khám phá dữ liệu.
- Chương 4 trình bày phương pháp đề xuất, kiến trúc hệ thống, pipeline xử lý dữ liệu và chi tiết từng mô hình.
- Chương 5 trình bày thực nghiệm, thiết lập đánh giá, kết quả trên MovieLens và Letterboxd, phân tích định lượng và định tính.
- Chương 6 tổng kết kết quả đạt được, hạn chế và hướng phát triển tiếp theo.
