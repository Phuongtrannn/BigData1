"""
01_etl_pipeline.py - THÀNH VIÊN 2: PHƯƠNG (DATA ENGINEER & ANALYST)
Dự án: Hệ Khuyến Nghị Sản Phẩm Thương Mại Điện Tử (RetailRocket Dataset)
Môn học: Nhập môn Big Data

Mục tiêu script:
1. Kết nối cụm phân tán Apache Spark (spark-master:7077) và Hadoop HDFS (hadoop-master:9000).
2. Nạp dữ liệu thô (raw events.csv) từ HDFS.
3. Tiền xử lý dữ liệu lớn bằng Spark DataFrame & RDD:
   - Loại bỏ trùng lặp (Deduplication).
   - Kiểm định tính toàn vẹn (Xử lý null, validate event_type hợp lệ).
   - Chuyển đổi định dạng timestamp từ Unix epoch (milliseconds) sang Spark TimestampType.
   - Ép kiểu dữ liệu tối ưu bộ nhớ (visitorid, itemid sang bigint/int).
4. Xuất dữ liệu sạch định dạng Parquet nén Snappy lưu trên HDFS phân tán và bản sao lưu cục bộ.
"""

import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp, count, isnan, when

def create_spark_session():
    """Khởi tạo SparkSession kết nối tới Cụm Spark Master hoặc Local fallback"""
    master_url = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    
    # Kiểm tra nếu chạy trực tiếp ngoài host không có master container thì dùng local[*]
    builder = SparkSession.builder \
        .appName("RetailRocket_ETL_Pipeline_Phuong") \
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
    print("🚀 BẮT ĐẦU LUỒNG TIỀN XỬ LÝ DỮ LIỆU ETL PHÂN TÁN (SPARK RDD / DATAFRAME)")
    print("=" * 70)

    spark = create_spark_session()

    # 1. Đường dẫn dữ liệu HDFS và Local Fallback
    hdfs_namenode = os.getenv("HDFS_NAMENODE", "hdfs://hadoop-master:9000")
    hdfs_input_path = f"{hdfs_namenode}/product_recommender/raw/events.csv"
    hdfs_output_path = f"{hdfs_namenode}/product_recommender/processed/events_clean.parquet"
    
    local_input_path = "E:/Bigdatadt3/Product_Recommender/data/raw/events.csv"
    if not os.path.exists(local_input_path):
        local_input_path = "data/raw/events.csv"
        
    local_output_path = "E:/Bigdatadt3/Product_Recommender/data/processed/events_clean.parquet"
    if not os.path.exists("E:/Bigdatadt3"):
        local_output_path = "data/processed/events_clean.parquet"

    # Ưu tiên đọc từ HDFS nếu chạy trong cụm
    input_source = hdfs_input_path
    output_target = hdfs_output_path

    print(f"\n[BƯỚC 1/5] Đang nạp dữ liệu thô từ HDFS: {input_source}")
    try:
        df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(input_source)
        df_raw.take(1)
        print("-> Đọc thành công dữ liệu từ HDFS NameNode!")
    except Exception as e:
        print(f"[WARN] Đọc từ HDFS không khả dụng ({e}). Sử dụng đường dẫn file cục bộ: {local_input_path}")
        df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(local_input_path)
        input_source = local_input_path
        output_target = local_output_path

    initial_count = df_raw.count()
    print(f"-> Tổng số bản ghi thô nạp vào Spark: {initial_count:,} dòng")
    print("-> Schema dữ liệu ban đầu:")
    df_raw.printSchema()

    # 2. Loại bỏ bản ghi trùng lặp (Deduplication)
    print("\n[BƯỚC 2/5] Tiến hành loại bỏ trùng lặp (Deduplication)...")
    df_dedup = df_raw.dropDuplicates(["timestamp", "visitorid", "event", "itemid"])
    dedup_count = df_dedup.count()
    duplicate_count = initial_count - dedup_count
    print(f"-> Đã phát hiện và loại bỏ: {duplicate_count:,} dòng trùng lặp.")
    print(f"-> Số bản ghi sau khi lọc trùng: {dedup_count:,} dòng")

    # 3. Lọc sạch dữ liệu rác, xử lý giá trị khuyết thiếu (Data Quality Cleaning)
    print("\n[BƯỚC 3/5] Kiểm định chất lượng và lọc giá trị không hợp lệ...")
    valid_events = ["view", "addtocart", "transaction"]
    
    # Điều kiện hợp lệ: visitorid, itemid, timestamp không rỗng; event nằm trong 3 hành vi chuẩn
    df_clean = df_dedup.filter(
        col("visitorid").isNotNull() &
        col("itemid").isNotNull() &
        col("timestamp").isNotNull() &
        col("event").isin(valid_events)
    )

    # 4. Chuẩn hóa kiểu dữ liệu (Schema Transformation & Feature Extraction)
    print("\n[BƯỚC 4/5] Chuyển đổi định dạng timestamp & kiểu dữ liệu...")
    # timestamp là millisecond -> chia 1000 sang giây để chuyển thành TimestampType chuẩn
    df_transformed = df_clean.withColumn("event_time", to_timestamp(from_unixtime(col("timestamp") / 1000))) \
                             .withColumn("visitorid", col("visitorid").cast("long")) \
                             .withColumn("itemid", col("itemid").cast("long")) \
                             .select("visitorid", "itemid", "event", "event_time", "transactionid")

    final_clean_count = df_transformed.count()
    print(f"-> Tổng số bản ghi đạt chuẩn chất lượng: {final_clean_count:,} dòng ({(final_clean_count/initial_count)*100:.2f}%)")
    print("-> Cấu trúc bảng Cleaned Event Schema:")
    df_transformed.printSchema()
    
    print("-> 10 dòng dữ liệu mẫu sau khi làm sạch:")
    df_transformed.show(10, truncate=False)

    # 5. Lưu trữ kết quả phân tán định dạng Parquet (Snappy Compressed)
    print(f"\n[BƯỚC 5/5] Ghi dữ liệu Parquet phân tán ra HDFS tại: {output_target}")
    df_transformed.write.mode("overwrite").parquet(output_target)
    print("-> Đã lưu thành công dữ liệu làm sạch vào HDFS!")

    # Đồng thời lưu một bản parquet cục bộ để phục vụ phân tích EDA / Streamlit nhanh
    try:
        print(f"-> Lưu bản sao cục bộ tại: {local_output_path}...")
        os.makedirs(os.path.dirname(local_output_path), exist_ok=True)
        df_transformed.write.mode("overwrite").parquet(local_output_path)
        print("-> Lưu bản sao cục bộ thành công!")
    except Exception as e:
        print(f"[NOTE] Bỏ qua ghi local do chạy hoàn toàn trong container: {e}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"✅ HOÀN THÀNH PIPELINE TIỀN XỬ LÝ ETL TRONG {elapsed} GIÂY")
    print("=" * 70)

    spark.stop()

if __name__ == "__main__":
    main()
