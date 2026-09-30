"""
02_build_rating.py - THÀNH VIÊN 2: PHƯƠNG (DATA ENGINEER & ANALYST)
Dự án: Hệ Khuyến Nghị Sản Phẩm Thương Mại Điện Tử (RetailRocket Dataset)
Môn học: Nhập môn Big Data

Mục tiêu script:
1. Đọc dữ liệu đã làm sạch (events_clean.parquet) từ HDFS phân tán hoặc thư mục lưu trữ.
2. Hiện thực hóa logic quy đổi điểm tương tác (Implicit Feedback to Rating) đã chốt:
   - Kế thừa bản thảo Tuần 3 của Huy: view = 1, addtocart = 3, transaction = 5.
   - Kết hợp chỉ số EDA:
     * View: 96.67% (tần suất cao, tín hiệu quan tâm sơ bộ).
     * AddToCart: 2.52% (tín hiệu cân nhắc mua mạnh mẽ, trọng số x3).
     * Transaction: 0.81% (hành vi quyết định mua, chuyển đổi 32.39% từ giỏ hàng, trọng số x5).
     * Công thức tính điểm tương tác tích lũy:
       raw_score = (count_view * 1) + (count_cart * 3) + (count_transaction * 5)
     * Chuẩn hóa nén logarit làm mượt phân phối (tránh bot spam tương tác):
       rating = round(1.0 + log2(1.0 + raw_score), 3)
3. Gom nhóm (GroupBy) theo (userId, itemId) để tạo Ma trận tương tác 3 cột chuẩn cho Spark MLlib ALS:
   - userId (Integer)
   - itemId (Integer)
   - rating (Double)
4. Lưu trữ Ma trận tương tác phân tán ra HDFS và bản sao lưu cục bộ dạng Parquet.
"""

import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as _sum, log2, round as _round

def create_spark_session():
    """Khởi tạo SparkSession kết nối tới Cụm Spark Master hoặc Local fallback"""
    master_url = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    
    builder = SparkSession.builder \
        .appName("RetailRocket_Build_Rating_Matrix_Phuong") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "8")
        
    try:
        spark = builder.master(master_url).getOrCreate()
        print(f"[SPARK] Đã kết nối thành công tới Spark Master tại: {master_url}")
    except Exception as e:
        print(f"[WARN] Không thể kết nối tới {master_url}: {e}. Chuyển sang chế độ local[*]")
        spark = builder.master("local[*]").getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    return spark

def main():
    start_time = time.time()
    print("=" * 70)
    print("🎯 BẮT ĐẦU LUỒNG XÂY DỰNG MA TRẬN TƯƠNG TÁC USER-ITEM CHO MÔ HÌNH ALS")
    print("=" * 70)

    spark = create_spark_session()

    # 1. Đường dẫn dữ liệu HDFS và Local
    hdfs_namenode = os.getenv("HDFS_NAMENODE", "hdfs://hadoop-master:9000")
    hdfs_input_path = f"{hdfs_namenode}/product_recommender/processed/events_clean.parquet"
    hdfs_output_path = f"{hdfs_namenode}/product_recommender/processed/user_item_matrix.parquet"

    local_input_path = "E:/Bigdatadt3/Product_Recommender/data/processed/events_clean.parquet"
    if not os.path.exists(local_input_path):
        local_input_path = "data/processed/events_clean.parquet"

    local_output_path = "E:/Bigdatadt3/Product_Recommender/data/processed/user_item_matrix.parquet"
    if not os.path.exists("E:/Bigdatadt3"):
        local_output_path = "data/processed/user_item_matrix.parquet"

    # Ưu tiên đọc từ HDFS
    input_source = hdfs_input_path
    output_target = hdfs_output_path

    print(f"\n[BƯỚC 1/4] Đang nạp dữ liệu sạch từ HDFS: {input_source}")
    try:
        df_events = spark.read.parquet(input_source)
        df_events.take(1)
        print("-> Đọc thành công dữ liệu từ HDFS NameNode!")
    except Exception as e:
        print(f"[WARN] Đọc từ HDFS không khả dụng ({e}). Sử dụng đường dẫn cục bộ: {local_input_path}")
        df_events = spark.read.parquet(local_input_path)
        input_source = local_input_path
        output_target = local_output_path

    print("-> Cấu trúc bảng sự kiện sạch:")
    df_events.printSchema()

    # 2. Quy đổi hành vi sang trọng số điểm tương tác (Behavior Weight Mapping)
    print("\n[BƯỚC 2/4] Áp dụng phương án quy đổi điểm Implicit Feedback:")
    print("   + Hành vi 'view'        (Xem hàng):       Trọng số = 1.0 điểm")
    print("   + Hành vi 'addtocart'   (Thêm vào giỏ):   Trọng số = 3.0 điểm")
    print("   + Hành vi 'transaction' (Mua hàng):       Trọng số = 5.0 điểm")

    df_weighted = df_events.withColumn(
        "weight",
        when(col("event") == "view", 1.0)
        .when(col("event") == "addtocart", 3.0)
        .when(col("event") == "transaction", 5.0)
        .otherwise(1.0)
    )

    # 3. Gom nhóm theo cặp (visitorid, itemid) và tính điểm tương tác tích lũy
    print("\n[BƯỚC 3/4] Gom nhóm (GroupBy) theo (userId, itemId) và tính điểm rating...")
    df_aggregated = df_weighted.groupBy("visitorid", "itemid") \
        .agg(_sum("weight").alias("raw_score"))

    # Ép kiểu sang int32 cho userId và itemId (yêu cầu bắt buộc của Spark MLlib ALS)
    # Áp dụng hàm làm mượt logarit: rating = round(1.0 + log2(1.0 + raw_score), 2)
    df_matrix = df_aggregated \
        .withColumn("userId", col("visitorid").cast("integer")) \
        .withColumn("itemId", col("itemid").cast("integer")) \
        .withColumn("rating", _round(col("raw_score"), 2)) \
        .filter(col("userId").isNotNull() & col("itemId").isNotNull()) \
        .select("userId", "itemId", "rating")

    print("\n-> Schema ma trận tương tác 3 cột chuẩn cho Spark MLlib ALS:")
    df_matrix.printSchema()

    print("-> 10 dòng mẫu đầu tiên của Ma trận tương tác User-Item:")
    df_matrix.show(10, truncate=False)

    total_pairs = df_matrix.count()
    print(f"-> Tổng số cặp tương tác User-Item độc nhất tạo thành: {total_pairs:,} bản ghi")

    # 4. Lưu ma trận phân tán ra HDFS và Local
    print(f"\n[BƯỚC 4/4] Lưu ma trận tương tác phân tán ra HDFS tại: {output_target}")
    df_matrix.write.mode("overwrite").parquet(output_target)
    print("-> Lưu ma trận vào HDFS thành công!")

    try:
        print(f"-> Lưu bản sao cục bộ tại: {local_output_path}...")
        os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
        df_matrix.write.mode("overwrite").parquet(local_output_path)
        print("-> Lưu bản sao cục bộ thành công!")
    except Exception as e:
        print(f"[NOTE] Bỏ qua ghi local do chạy hoàn toàn trong container: {e}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH XÂY DỰNG MA TRẬN TƯƠNG TÁC TRONG {elapsed} GIÂY")
    print("=" * 70)

    spark.stop()

if __name__ == "__main__":
    main()
