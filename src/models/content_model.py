import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFRecommender:
    def __init__(self):
        # Khởi tạo vectorizer, loại bỏ các từ vô nghĩa tiếng Anh (stop words)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.similarity_matrix = None
        self.movies_df = None

    def fit(self, movies_df):
        """
        Hàm này huấn luyện mô hình và lưu lại ma trận tương đồng.
        """
        self.movies_df = movies_df.reset_index(drop=True)
        
        # Bước 1: Gộp toàn bộ metadata thành một khối văn bản duy nhất
        # Yêu cầu cột content_text bao gồm: title + genres + tags + overview
        self.movies_df['content_text'] = (
            self.movies_df['title'].fillna('') + " " + 
            self.movies_df['genres'].str.replace('|', ' ').fillna('') + " " + 
            self.movies_df['tags'].fillna('') + " " + 
            self.movies_df['overview'].fillna('')
        )

        # Bước 2: Chuyển đổi văn bản thành ma trận TF-IDF
        print("Đang tính toán ma trận TF-IDF...")
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['content_text'])

        # Bước 3: Tính toán độ tương đồng Cosine giữa tất cả các phim
        print("Đang tính toán Cosine Similarity...")
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Bước 4: Lưu kết quả ra file .npy để Backend API tái sử dụng
        # Sử dụng '../artifacts' để thoát ra khỏi notebooks và trỏ đúng về thư mục gốc
        artifact_path = '../artifacts'
        os.makedirs(artifact_path, exist_ok=True)
        
        file_path = f'{artifact_path}/content_similarity.npy'
        np.save(file_path, self.similarity_matrix)
        print(f"Đã lưu ma trận tương đồng tại {file_path}")

    def recommend_similar_movies(self, movie_id, top_k=10):
        """
        Hàm gợi ý phim tương tự dựa trên 1 movie_id cụ thể.
        """
        if movie_id not in self.movies_df['movie_id'].values:
            return "Không tìm thấy phim trong cơ sở dữ liệu."

        # Lấy index (vị trí dòng) của bộ phim
        idx = self.movies_df.index[self.movies_df['movie_id'] == movie_id].tolist()[0]

        # Lấy mảng điểm tương đồng của phim này với tất cả phim khác
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sắp xếp danh sách theo điểm giảm dần
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Lấy top_k phim (bỏ qua chính nó ở vị trí index 0)
        sim_scores = sim_scores[1:top_k+1]
        
        # Trích xuất index
        movie_indices = [i[0] for i in sim_scores]

        # Trả về kết quả
        return self.movies_df.iloc[movie_indices][['movie_id', 'title', 'genres']]

    def recommend_content_for_user(self, user_id, user_history_df, top_k=10):
        """
        Hàm gợi ý phim dựa trên lịch sử tương tác (những phim đã thích/xem) của người dùng.
        """
        # Lấy danh sách movie_id mà user đã xem
        user_movies = user_history_df[user_history_df['user_id'] == user_id]['movie_id'].tolist()

        if not user_movies:
            return "Người dùng chưa có lịch sử xem phim (Cold-start)"

        # Lấy index của các bộ phim user đã xem
        watched_indices = self.movies_df.index[self.movies_df['movie_id'].isin(user_movies)].tolist()

        # Tính toán hồ sơ người dùng (User Profile) bằng cách tính trung bình cộng 
        # các vector tương đồng của những phim họ đã xem
        user_profile = np.mean(self.similarity_matrix[watched_indices], axis=0)

        # Sắp xếp điểm số từ hồ sơ người dùng
        sim_scores = list(enumerate(user_profile))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Lấy top_k phim nhưng phải LỌC BỎ những phim user đã xem rồi
        recommendations = [i for i in sim_scores if i[0] not in watched_indices][:top_k]
        movie_indices = [i[0] for i in recommendations]

        return self.movies_df.iloc[movie_indices][['movie_id', 'title', 'genres']]