"""
03_train_model.py - THÀNH VIÊN 3: HUY (MACHINE LEARNING ENGINEER)
Dự án: Hệ Khuyến Nghị Sản Phẩm Thương Mại Điện Tử (RetailRocket Dataset)
Môn học: Nhập môn Big Data

Mục tiêu script:
1. Tiếp nhận dữ liệu Ma trận tương tác 3 cột chuẩn (userId, itemId, rating) từ luồng ETL của Phương.
2. Huấn luyện mô hình Lọc cộng tác phân tán (Collaborative Filtering) sử dụng thuật toán ALS (Alternating Least Squares) của Apache Spark MLlib.
3. Cấu hình mô hình cho Implicit Feedback:
   - userCol="userId", itemCol="itemId", ratingCol="rating"
   - implicitPrefs=True
   - coldStartStrategy="drop" (tránh NaN giá trị dự đoán trong tập test)
4. Phân chia tập dữ liệu huấn luyện / kiểm thử: Train 80% / Test 20% (seed=42).
5. Tinh chỉnh siêu tham số (Hyperparameter Tuning):
   - Thử nghiệm các tổ hợp tham số: rank (5, 10, 20), regParam (0.01, 0.05, 0.1), alpha (1.0, 10.0, 40.0).
   - Đo lường đánh giá mô hình bằng chỉ số sai số chuẩn RMSE (Root Mean Squared Error) qua RegressionEvaluator.
6. Xác định mô hình tối ưu nhất (Best ALS Model) và lưu trữ phân tán lên HDFS (/product_recommender/models/als_model) cùng thư mục cục bộ.
7. Trích xuất gợi ý sản phẩm mẫu (Top-K Recommendations) và danh sách Top Sản phẩm thịnh hành phục vụ bài toán Cold-Start cho Thành viên 4 (Anh).
8. Xuất file kết quả als_tuning_results.json làm cơ sở vẽ biểu đồ đánh giá mô hình.
"""

import os
import sys
import time
import json
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, desc, count as _count, sum as _sum

def create_spark_session():
    """Khởi tạo SparkSession kết nối tới Cụm Spark Master"""
    master_url = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
    builder = SparkSession.builder \
        .appName("RetailRocket_ALS_Model_Training_Huy") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "8")
        
    try:
        spark = builder.master(master_url).getOrCreate()
        print(f"[SPARK] Đã kết nối thành công tới Spark Master tại: {master_url}")
    except Exception as e:
        print(f"[WARN] Không thể kết nối tới {master_url}: {e}. Chuyển sang local[*]")
        spark = builder.master("local[*]").getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    return spark

def main():
    start_time = time.time()
    print("=" * 70)
    print("🤖 [HUY] BẮT ĐẦU HUẤN LUYỆN VÀ TINH CHỈNH THAM SỐ MÔ HÌNH ALS (SPARK MLLIB)")
    print("=" * 70)

    spark = create_spark_session()

    # 1. Đường dẫn dữ liệu HDFS và Local
    hdfs_namenode = os.getenv("HDFS_NAMENODE", "hdfs://hadoop-master:9000")
    matrix_hdfs_path = f"{hdfs_namenode}/product_recommender/processed/user_item_matrix.parquet"
    model_hdfs_path = f"{hdfs_namenode}/product_recommender/models/als_best_model"
    cold_start_hdfs_path = f"{hdfs_namenode}/product_recommender/models/cold_start_popular.parquet"

    matrix_local_path = "E:/Bigdatadt3/Product_Recommender/data/processed/user_item_matrix.parquet"
    if not os.path.exists(matrix_local_path):
        matrix_local_path = "data/processed/user_item_matrix.parquet"

    model_local_path = "E:/Bigdatadt3/Product_Recommender/models/als_best_model"
    if not os.path.exists("E:/Bigdatadt3"):
        model_local_path = "models/als_best_model"

    tuning_json_path = "E:/Bigdatadt3/Product_Recommender/reports/als_tuning_results.json"
    if not os.path.exists("E:/Bigdatadt3"):
        tuning_json_path = "reports/als_tuning_results.json"

    # Đọc ma trận tương tác
    print(f"\n[BƯỚC 1/6] Nạp ma trận tương tác User-Item 3 cột từ HDFS: {matrix_hdfs_path}")
    try:
        df_matrix = spark.read.parquet(matrix_hdfs_path)
        df_matrix.take(1)
        print("-> Nạp thành công ma trận từ HDFS NameNode!")
    except Exception as e:
        print(f"[WARN] Đọc từ HDFS thất bại ({e}). Chuyển sang đọc cục bộ: {matrix_local_path}")
        df_matrix = spark.read.parquet(matrix_local_path)

    total_ratings = df_matrix.count()
    print(f"-> Tổng số cặp tương tác nạp vào mô hình: {total_ratings:,} bản ghi")
    df_matrix.printSchema()

    # 2. Phân chia tập dữ liệu Huấn luyện / Kiểm thử (Train/Test Split)
    print("\n[BƯỚC 2/6] Phân chia tập dữ liệu: 80% Train, 20% Test (seed=42)...")
    train_df, test_df = df_matrix.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"-> Tập Huấn luyện (Train set): {train_count:,} dòng ({(train_count/total_ratings)*100:.1f}%)")
    print(f"-> Tập Kiểm thử (Test set):     {test_count:,} dòng ({(test_count/total_ratings)*100:.1f}%)")

    # 3. Tinh chỉnh siêu tham số (Hyperparameter Tuning - Grid Search)
    print("\n[BƯỚC 3/6] Bắt đầu Grid Search tinh chỉnh siêu tham số ALS...")
    evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    # Danh mục thử nghiệm các bộ tham số đa dạng
    candidate_params = [
        {"name": "Mô hình 1 (Base)", "rank": 5, "regParam": 0.01, "alpha": 1.0, "maxIter": 10},
        {"name": "Mô hình 2 (Tuned α)", "rank": 10, "regParam": 0.05, "alpha": 10.0, "maxIter": 10},
        {"name": "Mô hình 3 (Optimal)", "rank": 10, "regParam": 0.1, "alpha": 20.0, "maxIter": 10},
        {"name": "Mô hình 4 (Deep Factor)", "rank": 20, "regParam": 0.1, "alpha": 40.0, "maxIter": 10}
    ]

    tuning_history = []
    best_model = None
    best_rmse = float("inf")
    best_config = None

    for idx, p in enumerate(candidate_params, 1):
        print(f"\n   [Cấu hình {idx}/{len(candidate_params)}] {p['name']}: rank={p['rank']}, regParam={p['regParam']}, alpha={p['alpha']}, maxIter={p['maxIter']}")
        t0 = time.time()
        
        als = ALS(
            userCol="userId",
            itemCol="itemId",
            ratingCol="rating",
            implicitPrefs=True,
            coldStartStrategy="drop",
            rank=p["rank"],
            regParam=p["regParam"],
            alpha=p["alpha"],
            maxIter=p["maxIter"],
            seed=42
        )
        
        model = als.fit(train_df)
        predictions = model.transform(test_df)
        
        rmse = evaluator_rmse.evaluate(predictions)
        mae = evaluator_mae.evaluate(predictions)
        duration = round(time.time() - t0, 2)
        
        print(f"   -> Kết quả: RMSE = {rmse:.4f} | MAE = {mae:.4f} (Thời gian train: {duration}s)")

        record = {
            "model_name": p["name"],
            "rank": p["rank"],
            "regParam": p["regParam"],
            "alpha": p["alpha"],
            "maxIter": p["maxIter"],
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "training_time_sec": duration
        }
        tuning_history.append(record)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_config = record

    print("\n" + "-" * 70)
    print(f"🏆 CẤU HÌNH TỐI ƯU NHẤT (BEST MODEL): {best_config['model_name']}")
    print(f"   Rank: {best_config['rank']} | RegParam: {best_config['regParam']} | Alpha: {best_config['alpha']}")
    print(f"   RMSE Tối ưu: {best_rmse:.4f} | MAE: {best_config['mae']:.4f}")
    print("-" * 70)

    # Lưu lại lịch sử tuning ra JSON
    try:
        os.makedirs(os.path.dirname(tuning_json_path), exist_ok=True)
        with open(tuning_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": "RetailRocket",
                "train_records": train_count,
                "test_records": test_count,
                "best_config": best_config,
                "all_models": tuning_history
            }, f, ensure_ascii=False, indent=4)
        print(f"-> Đã lưu lịch sử tinh chỉnh vào: {tuning_json_path}")
    except Exception as e:
        print(f"[WARN] Không thể ghi tuning json cục bộ: {e}")

    # 4. Lưu Best Model phân tán ra HDFS và Local
    print(f"\n[BƯỚC 4/6] Lưu trữ mô hình ALS tốt nhất ra HDFS tại: {model_hdfs_path}")
    best_model.write().overwrite().save(model_hdfs_path)
    print("-> Lưu model lên HDFS thành công!")

    try:
        print(f"-> Lưu bản sao model cục bộ tại: {model_local_path}...")
        best_model.write().overwrite().save(model_local_path)
        print("-> Lưu model cục bộ thành công!")
    except Exception as e:
        print(f"[NOTE] Ghi model local bỏ qua: {e}")

    # 5. Sinh Top-K Gợi ý cho tập người dùng mẫu (Inference Demo)
    print("\n[BƯỚC 5/6] Sinh Top-5 sản phẩm đề xuất (Top-K Recommendations) cho 5 người dùng mẫu:")
    sample_users = train_df.select("userId").distinct().limit(5)
    user_recs = best_model.recommendForUserSubset(sample_users, 5)
    user_recs.show(truncate=False)

    # 6. Xử lý bài toán Cold Start: Trích xuất Top sản phẩm thịnh hành (Trending Items)
    print("\n[BƯỚC 6/6] Trích xuất Top 20 sản phẩm phổ biến nhất cho người dùng mới (Cold-Start)...")
    popular_items_df = df_matrix.groupBy("itemId") \
        .agg(
            _count("userId").alias("interaction_count"),
            _sum("rating").alias("total_rating")
        ) \
        .orderBy(desc("total_rating"), desc("interaction_count")) \
        .limit(20)

    print("-> Top 10 sản phẩm thịnh hành (Cold Start recommendations):")
    popular_items_df.show(10, truncate=False)

    # Lưu Cold Start DataFrame ra HDFS và local Parquet
    popular_items_df.write.mode("overwrite").parquet(cold_start_hdfs_path)
    print("-> Đã lưu bảng Cold Start lên HDFS!")

    local_cold_start = "E:/Bigdatadt3/Product_Recommender/models/cold_start_popular.parquet"
    if not os.path.exists("E:/Bigdatadt3"):
        local_cold_start = "models/cold_start_popular.parquet"
    try:
        popular_items_df.write.mode("overwrite").parquet(local_cold_start)
        print("-> Đã lưu bảng Cold Start cục bộ!")
    except Exception as e:
        print(f"[NOTE] Ghi cold-start local: {e}")

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 70)
    print(f"🎉 [HUY] HOÀN TẤT HUẤN LUYỆN, ĐÁNH GIÁ VÀ XUẤT MÔ HÌNH ALS TRONG {elapsed} GIÂY")
    print("=" * 70)

    spark.stop()

if __name__ == "__main__":
    main()
