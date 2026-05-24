from data_loader_1M import MovieLensDataLoader
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent

    # Khởi tạo loader cấu hình đường dẫn thư mục nguồn (raw) và đích (processed)
    loader = MovieLensDataLoader(data_dir=data_dir / "raw", processed_dir=data_dir / "processed")
    
    # Chạy quy trình ETL khép kín: Đọc từ raw -> Đổ sạch ra file vật lý trong folder processed
    bundle = loader.load_and_save()
    
    # Phân chia dữ liệu train/val/test và tự động lưu luôn vào folder processed
    # train, val, test = loader.train_val_test_split(bundle.ratings, val_ratio=0.1, test_ratio=0.1, save_splits=True)
    
    print("\nHOÀN THÀNH PIPELINE CLEAN DATA!")
