import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as _sum

def main():
    # 1. Khởi tạo kết nối SparkSession tới Cụm Spark Master
    spark = SparkSession.builder \
        .appName("Implicit_Rating_Matrix_Builder") \
        .config("spark.master", "local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # 2. Định nghĩa đường dẫn dữ liệu
    hdfs_url = "hdfs://namenode:9000"
    
    # Đọc từ dữ liệu làm sạch (Parquet)
    input_clean_path = f"{hdfs_url}/data/events_clean.parquet"
    output_matrix_path = f"{hdfs_url}/data/als_implicit_ratings"
    
    # Tự động chuyển sang đường dẫn local nếu chạy trực tiếp bên ngoài HDFS container
    local_input_path = "data/events_clean.parquet"
    if os.path.exists(local_input_path) or os.path.exists("events_clean.parquet"):
        input_clean_path = local_input_path if os.path.exists(local_input_path) else "events_clean.parquet"
        output_matrix_path = "data/als_implicit_ratings"

    print(f"\n[1/4] Đang đọc dữ liệu đầu vào làm sạch (Parquet): {input_clean_path}")
    df_events = spark.read.parquet(input_clean_path)
    print("-> Đọc dữ liệu thành công! Cấu trúc dữ liệu:")
    df_events.printSchema()

    # 3. Viết logic quy đổi hành vi (Implicit Feedback) thành điểm số (Rating Score)
    # Rules: view = 1 điểm, addtocart = 3 điểm, transaction = 5 điểm
    print("\n[2/4] Áp dụng logic quy đổi điểm tương tác người dùng:")
    print("   - view = 1 điểm")
    print("   - addtocart = 3 điểm")
    print("   - transaction = 5 điểm")

    df_rated = df_events.withColumn(
        "event_score",
        when(col("event") == "view", 1)
        .when(col("event") == "addtocart", 3)
        .when(col("event") == "transaction", 5)
        .otherwise(1)
    )

    # 4. GroupBy theo (userId, itemId) và tính tổng điểm để tạo ra tập dữ liệu 3 cột chuẩn cho ALS
    print("\n[3/4] Gom nhóm (GroupBy) và tính tổng điểm tương tác tích lũy (3 cột chuẩn)...")
    df_user_item_matrix = df_rated.groupBy("visitorid", "itemid") \
        .agg(_sum("event_score").alias("implicit_rating")) \
        .withColumnRenamed("visitorid", "userId") \
        .withColumnRenamed("itemid", "itemId")

    print("\n-> Cấu trúc Bảng tương tác User-Item 3 cột chuẩn cho mô hình Spark ALS:")
    df_user_item_matrix.printSchema()
    
    print("-> Xem thử 10 dòng dữ liệu ma trận đầu tiên:")
    df_user_item_matrix.show(10, truncate=False)

    total_records = df_user_item_matrix.count()
    print(f"-> Tổng số cặp tương tác User-Item tạo ra: {total_records:,} bản ghi")

    # 5. Lưu ma trận 3 cột dạng phân tán Parquet ra HDFS / Local
    print(f"\n[4/4] Đang lưu ma trận 3 cột ra tại: {output_matrix_path}")
    df_user_item_matrix.write.mode("overwrite").parquet(output_matrix_path)
    print("-> Lưu ma trận thành công!")

    spark.stop()

if __name__ == "__main__":
    main()
