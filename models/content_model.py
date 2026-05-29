import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# =====================================================================
# FILE: content_model.py
# CHỨC NĂNG: Đây là "Nhà máy trích xuất đặc trưng" (Feature Extractor).
# Nhiệm vụ của nó là đọc các thông tin chữ viết của bộ phim (tên, thể loại, cốt truyện)
# và biến đổi chúng thành các ma trận số học (vector) để máy tính có thể hiểu được.
# =====================================================================

class TFIDFRecommender:
    """
    CLASS 1: Mô hình TF-IDF (Term Frequency-Inverse Document Frequency)
    - Vai trò: Đây là mô hình cơ sở (Baseline).
    - Cách hoạt động: Đếm số lần xuất hiện của các từ. Từ nào hiếm (đặc trưng) thì điểm cao.
    - Ưu điểm: Chạy cực kỳ nhanh, tốn ít RAM, không cần GPU.
    - Nhược điểm: Chỉ hiểu được mặt chữ, không hiểu được ngữ nghĩa sâu xa.
    """
    def __init__(self):
        """Khởi tạo cấu hình cho TF-IDF"""
        print("[TF-IDF] Đã khởi tạo công cụ TfidfVectorizer.")
        # stop_words='english': Tự động loại bỏ các từ vô nghĩa trong tiếng Anh (như 'the', 'is', 'in', 'and'...) 
        # vì chúng xuất hiện nhiều nhưng không mang lại đặc trưng cho bộ phim.
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def extract_and_save(self, df, artifact_path='artifacts'):
        print("[TF-IDF] Đang xử lý văn bản...")
        
        # BƯỚC 1: GỘP VĂN BẢN (Text Aggregation)
        # Chúng ta ghép Tên phim + Thể loại + Cốt truyện thành một đoạn văn duy nhất.
        # Dùng .fillna('') để tránh lỗi nếu có phim bị thiếu thông tin (giá trị NaN).
        # Dùng .str.replace('|', ' ') để biến "Action|Sci-Fi" thành "Action Sci-Fi".
        content_text = df['title'].fillna('') + " " + \
                       df['genres'].str.replace('|', ' ').fillna('') + " " + \
                       df['overview'].fillna('')
        
        print("[TF-IDF] Đang trích xuất ma trận đặc trưng...")
        
        # BƯỚC 2: BIẾN ĐỔI CHỮ THÀNH SỐ (Vectorization)
        # Lệnh fit_transform sẽ học từ vựng từ tất cả các phim và tạo ra một "ma trận thưa" (sparse matrix).
        # (Ma trận thưa là ma trận chứa rất nhiều số 0 để tiết kiệm RAM).
        tfidf_matrix = self.vectorizer.fit_transform(content_text)
        
        # BƯỚC 3: LƯU TRỮ VÀ ĐÓNG GÓI (Exporting)
        os.makedirs(artifact_path, exist_ok=True) # Tạo thư mục 'artifacts' nếu chưa có
        
        # Tạo Từ điển ánh xạ (Mapping ID):
        # Lý do: Trong ma trận Numpy, các hàng được đánh số 0, 1, 2...
        # Nhưng mã phim (movieId) trong file CSV lại là 1, 2, 5, 10 (nhảy cóc).
        # Từ điển này giống như "danh bạ" giúp ta biết: movieId = 5 đang nằm ở hàng số mấy trong ma trận.
        movie_id_map = {row['movieId']: idx for idx, row in df.iterrows()}
        np.save(f'{artifact_path}/movie_id_map.npy', movie_id_map)
        
        # Lưu ý: TF-IDF mặc định trả về ma trận thưa (để nhẹ máy). 
        # Nhưng để đồng bộ với cấu trúc Dense (đặc) của Pytorch và SBERT sau này, 
        # ta phải ép nó về ma trận đặc bằng lệnh .toarray() trước khi lưu.
        np.save(f'{artifact_path}/tfidf_embeddings.npy', tfidf_matrix.astype(np.float32).toarray())
        
        print(f"[TF-IDF] Đã lưu xong vector tại {artifact_path}/tfidf_embeddings.npy")
        print(f"[TF-IDF] Chiều dữ liệu: {tfidf_matrix.shape}")


class SBERTExtractor:
    """
    CLASS 2: Mô hình SBERT (Sentence-BERT)
    - Vai trò: Đây là mô hình Học Sâu Nâng Cao (Deep Learning) - Trái tim của dự án.
    - Cách hoạt động: Dùng mạng Neural để đọc hiểu ngữ nghĩa của cả câu văn.
    - Ưu điểm: Rất thông minh, hiểu được từ đồng nghĩa (Ví dụ: 'Alien' và 'Space monster' là giống nhau).
    - Nhược điểm: Chạy chậm hơn và cần nhiều RAM máy tính hơn TF-IDF.
    """
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Khởi tạo và tải mạng nơ-ron ngôn ngữ"""
        print(f"[SBERT] Đang tải mạng nơ-ron ngôn ngữ {model_name}...")
        # all-mpnet-base-v2 là phiên bản tốt nhất hiện nay của SBERT cho việc ánh xạ câu thành vector 768-chiều.
        self.model = SentenceTransformer(model_name)

    def extract_and_save(self, df, artifact_path='artifacts'):
        print("[SBERT] Đang chuẩn bị văn bản ngữ nghĩa...")
        
        # Tương tự như TF-IDF, ta cũng gộp tất cả chữ nghĩa của phim thành 1 đoạn văn để AI đọc.
        content_text = df['title'].fillna('') + " " + \
                       df['genres'].str.replace('|', ' ').fillna('') + " " + \
                       df['overview'].fillna('')
        
        print("[SBERT] Bắt đầu mã hóa (Sẽ tốn thời gian, vui lòng chờ)...")
        
        # BƯỚC QUAN TRỌNG: MÃ HÓA NGỮ NGHĨA (Encoding)
        # Hàm encode() sẽ nhận các đoạn văn bản và dịch chúng thành ma trận số học.
        # batch_size=8: Bắt AI xử lý nhóm 8 bộ phim một lần. Chỉnh số này quá cao máy sẽ bị tràn RAM và sập.
        # show_progress_bar=True: Hiện thanh tải % để biết máy chưa bị treo.
        embeddings = self.model.encode(
            content_text.tolist(), 
            batch_size=8, 
            show_progress_bar=True
        )
        
        os.makedirs(artifact_path, exist_ok=True)
        
        # 1. Lưu ma trận vector SBERT (Mỗi bộ phim giờ là 1 vector gồm 768 con số thập phân)
        np.save(f'{artifact_path}/movie_embeddings.npy', embeddings)
        
        # 2. Lưu Từ điển ánh xạ ID (Giống hệt mục đích bên TF-IDF giải thích ở trên)
        movie_id_map = {row['movieId']: idx for idx, row in df.iterrows()}
        np.save(f'{artifact_path}/movie_id_map.npy', movie_id_map)
        
        print(f"[SBERT] Hoàn tất! Đã lưu vector 768-chiều tại {artifact_path}/movie_embeddings.npy")