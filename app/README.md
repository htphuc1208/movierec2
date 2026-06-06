# Phân hệ Web & API Đề xuất Phim

Tài liệu này mô tả phần giao diện người dùng (Frontend) và các API phục vụ đề xuất phim (Backend) của dự án. Hệ thống được thiết kế theo kiến trúc Client-Server, tích hợp trực tiếp với các mô hình AI đã huấn luyện.

### Note: Hiện tại đang sử dụng dữ liệu từ dataset ml-latest-small kết hợp với enriched_movies.csv

## Các tính năng đã hoàn thiện trong phân hệ này

1. **Giao diện Trang chủ (IMDb Style):**
   - Hero Banner nổi bật cho phim Top 1 Trending.
   - Băng chuyền (Carousel) vuốt ngang cho các luồng phim (Mới ra mắt, Hành động, Hài hước,...).
   - **A/B Testing AI:** Cho phép chuyển đổi trực tiếp giữa mô hình `LightGCN` (Đồ thị) và `SVD` (Ma trận) để so sánh kết quả gợi ý.

2. **Trang Chi tiết Phim:**
   - Hiển thị thông tin chi tiết (Poster, Điểm số, Thể loại, Tóm tắt).
   - Tích hợp **SBERT (NLP):** Tự động gợi ý các bộ phim có nội dung tương đồng.

3. **Quản lý Lịch sử User:**
   - Trang Lịch sử riêng biệt dạng lưới (Grid).
   - Hỗ trợ bộ lọc và sắp xếp (Theo điểm số, theo bảng chữ cái A-Z) xử lý trực tiếp trên Frontend giúp phản hồi tức thì.

4. **Thanh điều hướng (Navbar):**
   - Tìm kiếm phim (ấn Enter để chuyển trang kết quả).
   - Dropdown chọn User (để giả lập đăng nhập và lấy context gợi ý).

---

## 📂 Các file quan trọng

- `app/streamlit_app.py`: Chứa toàn bộ mã nguồn Frontend vẽ giao diện và logic chuyển trang.
- `api/main.py`: Chứa mã nguồn Backend (FastAPI) xử lý logic gọi Model (`SVD`, `LightGCN`, `SBERT`) và giao tiếp với file dữ liệu CSV.

*(Lưu ý cho team: Đảm bảo các file trọng số mô hình đã được đặt đúng trong thư mục `artifacts/` và file CSV nằm trong `data/` trước khi chạy).*

---

## 🛠️ Hướng dẫn khởi chạy (Dành cho Dev)

Để chạy hệ thống trên máy cá nhân, mọi người cần mở **2 cửa sổ Terminal** riêng biệt tại thư mục gốc của dự án.

### Bước 1: Khởi động Backend API
Mở Terminal 1 và chạy lệnh sau để bật server FastAPI (Cổng mặc định: 8000):
```bash
uvicorn api.main:app --reload
```
Lưu ý: Quá trình khởi động có thể mất khoảng 30 giây để server nạp mô hình SBERT vào bộ nhớ (RAM).

### Bước 2: Khởi động Frontend Giao diện
Mở Terminal 2 và chạy lệnh sau để bật giao diện web Streamlit:

```bash
streamlit run app/streamlit_app.py
```

Trình duyệt sẽ tự động mở lên tại địa chỉ `http://localhost:8501`. Mọi người có thể chọn User trên thanh Menu để bắt đầu test các luồng đề xuất AI!