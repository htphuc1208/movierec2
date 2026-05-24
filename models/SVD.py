from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError: 
    torch = None
    nn = None


if torch is not None:

    class SVDModel(nn.Module):
        """Funk SVD"""

        def __init__(
            self, 
            num_users: int, 
            num_items: int, 
            embedding_dim: int = 64, 
            global_mean: float = 0.0
        ) -> None:
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.embedding_dim = embedding_dim
            
            # Khởi tạo hằng số điểm trung bình toàn cục
            self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))

            # Ma trận nhúng đặc trưng ẩn
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)

            # Các tham số chệch để chuẩn hóa xu hướng chấm điểm
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)

            # Khởi tạo trọng số ngẫu nhiên theo phân phối chuẩn nhỏ
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

        def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
            """Dự đoán điểm rating cho một cặp user_ids và item_ids.
            
            Công thức: hat_y = global_mean + b_u + b_i + <P_u, Q_i>
            """
            # Lấy vector nhúng tương ứng
            p_u = self.user_embedding(user_ids)  
            q_i = self.item_embedding(item_ids)  

            # Lấy các giá trị bias tương ứng
            b_u = self.user_bias(user_ids).squeeze(1)  
            b_i = self.item_bias(item_ids).squeeze(1)  

            # Tính tích vô hướng giữa đặc trưng của User và Item
            dot_product = (p_u * q_i).sum(dim=1)  

            # Cộng toàn bộ lại để ra điểm số dự đoán
            rating_pred = self.global_mean + b_u + b_i + dot_product
            return rating_pred

else:

    class SVDModel: 
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("SVDModel requires torch. Install requirements-ml.txt.")