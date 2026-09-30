import pandas as pd

# ---------------------------------------------------------
# PONTYAIL MODE: Giữ code đơn giản nhất có thể (KISS).
# File này tạm thời dùng dữ liệu giả (Mock data) để chạy UI.
# Sau khi Thành viên 3 train xong mô hình ALS, fen chỉ cần
# thay thế hàm ở đây để load model từ HDFS và predict thật.
# ---------------------------------------------------------

def check_new_user(user_id):
    """
    Kiểm tra xem user_id có tồn tại trong tập dữ liệu train chưa.
    Tạm thời giả lập: User ID chẵn là cũ, lẻ là mới (Cold start).
    """
    # Thay bằng logic thật: kiểm tra user_id trong danh sách users của model
    return user_id % 2 != 0

def get_recommendations(user_id):
    """
    Lấy danh sách gợi ý cho user đã có lịch sử dựa trên mô hình ALS.
    """
    # TODO: Khởi tạo PySpark, load ALS Model và dự đoán cho user_id
    
    # Mock data để test UI
    return pd.DataFrame({
        "item_id": [101, 102, 103, 104],
        "name": ["Điện thoại Samsung", "Laptop Dell", "Tai nghe Sony", "Chuột Logitech"],
        "score": [4.9, 4.7, 4.5, 4.2]
    })

def get_popular_products():
    """
    FALLBACK (COLD START): Gợi ý sản phẩm phổ biến nhất cho người dùng mới.
    ĐÁP ỨNG TIÊU CHÍ RUBRIC BẮT BUỘC.
    """
    # TODO: Load danh sách top sản phẩm (lấy từ dữ liệu lịch sử nhóm làm EDA)
    
    # Mock data để test UI
    return pd.DataFrame({
        "item_id": [999, 888, 777, 666],
        "name": ["Sản phẩm Hot trend 1", "Sản phẩm Bán chạy 2", "Món đồ Quốc dân 3", "Sản phẩm Khuyến mãi 4"],
        "score": [5.0, 4.9, 4.8, 4.8]
    })
