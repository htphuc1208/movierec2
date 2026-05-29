import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. KIẾN TRÚC MỘT TÒA THÁP (TÁI SỬ DỤNG CHO CẢ USER TOWER VÀ ITEM TOWER)
# =====================================================================
class Tower(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        """
        input_dim: 768 chiều (Kích thước vector gốc từ mô hình ngôn ngữ SBERT).
        hidden_dim: 256 chiều (Lớp ẩn giúp mạng học các tổ hợp đặc trưng phức tạp phi tuyến tính).
        output_dim: 128 chiều (Không gian nhúng cuối cùng, nén lại cho nhẹ và dễ so sánh).
        """
        super(Tower, self).__init__()
        # Lớp tuyến tính thứ nhất (Thu gọn từ 768 xuống 256)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Hàm kích hoạt ReLU tạo tính phi tuyến giúp mạng nơ-ron học thông minh hơn
        self.relu = nn.ReLU()
        
        # Lớp Dropout (Tắt ngẫu nhiên 20% nơ-ron) chống học vẹt (Overfitting)
        self.dropout = nn.Dropout(0.2)
        
        # Lớp tuyến tính đầu ra (Thu gọn từ 256 xuống 128)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =====================================================================
# 2. KIẾN TRÚC TỔNG HAI THÁP ĐÔI (TWO-TOWER MODEL)
# =====================================================================
class TwoTowerModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super(TwoTowerModel, self).__init__()
        # Khởi tạo 2 tòa tháp độc lập: 1 cho User, 1 cho Phim
        # Mặc dù cấu trúc giống nhau, nhưng trọng số (weights) của chúng sẽ tự học và cập nhật khác nhau
        self.user_tower = Tower(input_dim, hidden_dim, output_dim)
        self.item_tower = Tower(input_dim, hidden_dim, output_dim)

    def forward(self, user_features, item_features):
        """
        Đầu vào:
        - user_features: Tensor vector profile của người dùng (batch_size, 768)
        - item_features: Tensor vector SBERT của phim (batch_size, 768)
        """
        # 1. Cho dữ liệu đặc trưng leo lên từng tòa tháp tương ứng
        u_emb = self.user_tower(user_features)
        i_emb = self.item_tower(item_features)
        
        # 2. Chuẩn hóa L2 (L2 Normalization)
        # BẮT BUỘC: Đưa tất cả vector về độ dài bằng 1. Lúc này phép tích vô hướng (Dot Product)
        # ở bước dưới sẽ hoàn toàn tương đương với Cosine Similarity.
        u_emb = F.normalize(u_emb, p=2, dim=1)
        i_emb = F.normalize(i_emb, p=2, dim=1)
        
        # 3. Đỉnh tháp: Tính điểm tương đồng bằng tích vô hướng
        # Điểm càng cao (>0) nghĩa là Phim càng hợp với User. Điểm thấp (<0) là không hợp.
        score = torch.sum(u_emb * i_emb, dim=1)
        
        return score

if __name__ == "__main__":
    print("Khởi tạo thử nghiệm mô hình Two-Tower...")
    model = TwoTowerModel()
    print(model)
    print("=> Tệp kiến trúc mạng nơ-ron đã sẵn sàng!")