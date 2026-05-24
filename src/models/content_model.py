import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================================
# MÔ HÌNH 1: BASELINE (TF-IDF) 
# Chức năng: Đóng vai trò làm mô hình cơ sở, chạy nhanh, làm phương án dự phòng.
# =====================================================================
class TFIDFRecommender:
    def __init__(self):
        # Khởi tạo công cụ biến đổi chữ thành số (Vectorizer).
        # stop_words='english': Tự động loại bỏ các từ tiếng Anh vô nghĩa (the, is, in, at...)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        """
        Hàm huấn luyện mô hình. Nhận đầu vào là bảng dữ liệu phim, xử lý văn bản,
        tính toán độ giống nhau giữa các phim và lưu file ra ổ cứng.
        """
        # Đặt lại index cho dataframe để đảm bảo thứ tự luôn khớp với ma trận toán học
        self.movies_df = movies_df.reset_index(drop=True)
        
        # BƯỚC 1: TẠO HỒ SƠ VĂN BẢN (CONTENT TEXT)
        # Gộp tất cả thông tin quan trọng của phim thành 1 chuỗi dài.
        # Dùng fillna('') để đề phòng trường hợp phim bị khuyết dữ liệu sẽ không bị báo lỗi.
        self.movies_df['content_text'] = (
            self.movies_df['title'].fillna('') + " " + 
            self.movies_df['genres'].str.replace('|', ' ').fillna('') + " " + 
            self.movies_df['tags'].fillna('') + " " + 
            self.movies_df['overview'].fillna('')
        )

        # BƯỚC 2: MÃ HÓA VĂN BẢN (VECTORIZATION)
        print("[TF-IDF] Đang tính toán ma trận TF-IDF...")
        # Biến cột văn bản thành một ma trận số học đếm tần suất từ vựng
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['content_text'])

        # BƯỚC 3: TÍNH ĐỘ TƯƠNG ĐỒNG (COSINE SIMILARITY)
        print("[TF-IDF] Đang tính toán Cosine Similarity...")
        # So sánh khoảng cách góc giữa tất cả các phim với nhau
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # BƯỚC 4: LƯU MÔ HÌNH
        # Thoát ra thư mục gốc và lưu vào thư mục 'artifacts'
        artifact_path = '../artifacts'
        os.makedirs(artifact_path, exist_ok=True) # Tự tạo thư mục nếu chưa có
        
        # Lưu ra tệp npy, đặt tên riêng để không ghi đè lên file của mô hình SBERT
        file_path = f'{artifact_path}/tfidf_similarity.npy'
        np.save(file_path, self.similarity_matrix)
        print(f"[TF-IDF] Đã lưu mô hình dự phòng tại {file_path}")

    def recommend_similar_movies(self, movie_id, top_k=10):
        """
        Gợi ý K bộ phim giống với bộ phim đang xem nhất.
        """
        # Kiểm tra xem phim có tồn tại trong dữ liệu không
        if movie_id not in self.movies_df['movie_id'].values:
            return "Không tìm thấy phim."
            
        # Tìm vị trí (index) của bộ phim này trong bảng dữ liệu
        idx = self.movies_df.index[self.movies_df['movie_id'] == movie_id].tolist()[0]
        
        # Lấy điểm tương đồng của phim này với tất cả phim khác, kết hợp với index
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sắp xếp điểm số từ cao xuống thấp. Bỏ qua vị trí 0 vì đó là chính nó (điểm = 1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        
        # Lọc lấy index và trả về thông tin phim tương ứng
        return self.movies_df.iloc[[i[0] for i in sim_scores]][['movie_id', 'title', 'genres']]

    def recommend_content_for_user(self, user_id, user_history_df, top_k=10):
        """
        Gợi ý phim dựa trên lịch sử tương tác của người dùng.
        """
        # Lấy danh sách ID các phim người dùng đã xem
        user_movies = user_history_df[user_history_df['user_id'] == user_id]['movie_id'].tolist()
        
        # Nếu chưa xem phim nào -> Gặp bài toán Khởi đầu lạnh (Cold-start)
        if not user_movies:
            return "Cold-start"
            
        # Lấy index của các bộ phim người dùng đã xem
        watched_indices = self.movies_df.index[self.movies_df['movie_id'].isin(user_movies)].tolist()
        
        # TẠO HỒ SƠ NGƯỜI DÙNG (USER PROFILE): 
        # Tính trung bình cộng ma trận của các phim đã xem để đại diện cho gu xem phim
        user_profile = np.mean(self.similarity_matrix[watched_indices], axis=0)
        
        # Sắp xếp các phim giống với gu này nhất
        sim_scores = sorted(list(enumerate(user_profile)), key=lambda x: x[1], reverse=True)
        
        # Lọc bỏ những phim đã xem ra khỏi danh sách gợi ý
        recommendations = [i for i in sim_scores if i[0] not in watched_indices][:top_k]
        
        return self.movies_df.iloc[[i[0] for i in recommendations]][['movie_id', 'title', 'genres']]


# =====================================================================
# MÔ HÌNH 2: NÂNG CAO (SBERT)
# Chức năng: Mô hình chính thức dùng AI học sâu để phân tích ngữ nghĩa 
# chuẩn bị đầu vào cho Kiến trúc Tháp Đôi (Two-Tower Model).
# =====================================================================
class SBERTRecommender:
    def __init__(self):
        # Tải mô hình ngôn ngữ SBERT (Bản đầy đủ: 768 chiều vector)
        print("[SBERT] Đang tải mô hình bản đầy đủ (all-mpnet-base-v2)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.similarity_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        """
        Hàm huấn luyện SBERT. Quá trình này sẽ tốn nhiều thời gian hơn do chạy trên CPU.
        """
        self.movies_df = movies_df.reset_index(drop=True)
        
        # BƯỚC 1: Gộp văn bản (Giống hệt TF-IDF)
        self.movies_df['content_text'] = (
            self.movies_df['title'].fillna('') + " " + 
            self.movies_df['genres'].str.replace('|', ' ').fillna('') + " " + 
            self.movies_df['tags'].fillna('') + " " + 
            self.movies_df['overview'].fillna('')
        )

        # BƯỚC 2: MÃ HÓA BẰNG HỌC SÂU (DEEP LEARNING ENCODING)
        print("[SBERT] Đang mã hóa văn bản (batch_size=8 để bảo vệ RAM 8GB)...")
        # Chia nhỏ dữ liệu ra thành từng cụm 8 phim để CPU không bị quá tải
        embeddings = self.model.encode(
            self.movies_df['content_text'].tolist(), 
            batch_size=8, 
            show_progress_bar=True 
        )

        # BƯỚC 3: TÍNH ĐỘ TƯƠNG ĐỒNG
        print("[SBERT] Đang tính toán Cosine Similarity...")
        self.similarity_matrix = cosine_similarity(embeddings, embeddings)

        # BƯỚC 4: LƯU TỆP BÀN GIAO CHO BACKEND
        artifact_path = '../artifacts'
        os.makedirs(artifact_path, exist_ok=True)
        
        # File 1: Lưu Vector 768 chiều để sau này dùng cho Kiến trúc Tháp Đôi
        np.save(f'{artifact_path}/movie_embeddings.npy', embeddings)
        print(f"[SBERT] Đã lưu ma trận nhúng (768 chiều) tại {artifact_path}/movie_embeddings.npy")
        
        # File 2: Lưu ma trận điểm số tương đồng để gợi ý trực tiếp ngay lập tức
        np.save(f'{artifact_path}/content_similarity.npy', self.similarity_matrix)
        print(f"[SBERT] Đã lưu ma trận tương đồng tại {artifact_path}/content_similarity.npy")

    def recommend_similar_movies(self, movie_id, top_k=10):
        """ Gợi ý phim giống nhau (Logic giống TF-IDF) """
        if movie_id not in self.movies_df['movie_id'].values:
            return "Không tìm thấy phim."
        idx = self.movies_df.index[self.movies_df['movie_id'] == movie_id].tolist()[0]
        sim_scores = sorted(list(enumerate(self.similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:top_k+1]
        return self.movies_df.iloc[[i[0] for i in sim_scores]][['movie_id', 'title', 'genres']]

    def recommend_content_for_user(self, user_id, user_history_df, top_k=10):
        """ Gợi ý phim cho người dùng dựa trên sở thích (Logic giống TF-IDF) """
        user_movies = user_history_df[user_history_df['user_id'] == user_id]['movie_id'].tolist()
        if not user_movies:
            return "Cold-start"
        watched_indices = self.movies_df.index[self.movies_df['movie_id'].isin(user_movies)].tolist()
        user_profile = np.mean(self.similarity_matrix[watched_indices], axis=0)
        sim_scores = sorted(list(enumerate(user_profile)), key=lambda x: x[1], reverse=True)
        recommendations = [i for i in sim_scores if i[0] not in watched_indices][:top_k]
        return self.movies_df.iloc[[i[0] for i in recommendations]][['movie_id', 'title', 'genres']]