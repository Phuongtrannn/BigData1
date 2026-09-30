import json

def create_eda_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ĐỒ ÁN BIG DATA: HỆ KHUYẾN NGHỊ SẢN PHẨM (RETAILROCKET DATASET)\n",
                    "## NOTEBOOK 01: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA) & TIỀN XỬ LÝ\n",
                    "**Thành viên thực hiện:** PHƯƠNG (Chuyên viên Phân tích & Kỹ sư Dữ liệu)\n",
                    "\n",
                    "### Nội dung thực hiện:\n",
                    "1. Nạp và khảo sát cấu trúc tập dữ liệu `events.csv`.\n",
                    "2. Phân tích phân bố hành vi (View, AddToCart, Transaction) & Phễu chuyển đổi (Conversion Funnel).\n",
                    "3. Phân tích mật độ tương tác theo chu kỳ thời gian (Giờ & Thứ trong tuần).\n",
                    "4. Đánh giá tính chất Long-tail của người dùng và sản phẩm.\n",
                    "5. Xác định Top 10 sản phẩm thịnh hành (Cold-Start candidates).\n",
                    "6. Xây dựng logic quy đổi phản hồi ẩn (Implicit Feedback) thành điểm số."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Thiết lập cấu hình trực quan hóa\n",
                    "sns.set_theme(style='whitegrid', palette='muted')\n",
                    "plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Nạp dữ liệu sự kiện tương tác RetailRocket"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "data_path = '../data/raw/events.csv'\n",
                    "df = pd.read_csv(data_path)\n",
                    "print(f'Tổng số dòng dữ liệu: {len(df):,}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Tiền xử lý trường thời gian và thống kê cơ bản"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')\n",
                    "df['hour'] = df['datetime'].dt.hour\n",
                    "df['day_of_week'] = df['datetime'].dt.day_name()\n",
                    "\n",
                    "print('Số người dùng unique:', df['visitorid'].nunique())\n",
                    "print('Số sản phẩm unique:', df['itemid'].nunique())\n",
                    "print('Khoảng thời gian:', df['datetime'].min(), '->', df['datetime'].max())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Phân bố các loại hành vi người dùng (Event Distribution)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "event_counts = df['event'].value_counts()\n",
                    "display(event_counts.to_frame('Số lượng'))\n",
                    "\n",
                    "plt.figure(figsize=(7, 4))\n",
                    "sns.barplot(x=event_counts.index, y=event_counts.values, palette='Blues_r', edgecolor='black')\n",
                    "plt.yscale('log')\n",
                    "plt.title('Phân bố loại hành vi người dùng (Thang Logarit)', fontweight='bold')\n",
                    "plt.ylabel('Số lượng (Log scale)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Phân tích tương tác theo khung giờ trong ngày"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "hourly = df.groupby(['hour', 'event']).size().unstack(fill_value=0)\n",
                    "plt.figure(figsize=(10, 4.5))\n",
                    "plt.plot(hourly.index, hourly['view'], label='View', marker='o', color='#3B82F6')\n",
                    "plt.title('Mật độ tương tác theo từng giờ trong ngày (24 Giờ)', fontweight='bold')\n",
                    "plt.xlabel('Khung giờ (UTC 0h - 23h)')\n",
                    "plt.ylabel('Số lượt xem')\n",
                    "plt.xticks(range(24))\n",
                    "plt.legend()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Hiện tượng Long-Tail & Bài toán Cold-Start"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "user_interactions = df.groupby('visitorid').size()\n",
                    "single_int_pct = (user_interactions == 1).mean() * 100\n",
                    "print(f'Tỷ lệ người dùng chỉ tương tác 1 lần (Cold-Start): {single_int_pct:.2f}%')\n",
                    "\n",
                    "# Top 10 sản phẩm thịnh hành nhất\n",
                    "top_items = df['itemid'].value_counts().head(10)\n",
                    "display(top_items.to_frame('Số lượt tương tác'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 6. Logic quy đổi điểm tương tác (Implicit Feedback Weighting)\n",
                    "Quy tắc gán trọng số: `view` = 1, `addtocart` = 3, `transaction` = 5."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "weights = {'view': 1, 'addtocart': 3, 'transaction': 5}\n",
                    "df['weight'] = df['event'].map(weights)\n",
                    "user_item_matrix = df.groupby(['visitorid', 'itemid'])['weight'].sum().reset_index()\n",
                    "user_item_matrix.columns = ['userId', 'itemId', 'rating']\n",
                    "print(f'Số cặp User-Item độc nhất: {len(user_item_matrix):,}')\n",
                    "user_item_matrix.head(10)"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("E:/Bigdatadt3/Product_Recommender/notebooks/01_EDA_Data.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("-> Đã tạo thành công notebooks/01_EDA_Data.ipynb")

def create_als_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ĐỒ ÁN BIG DATA: HỆ KHUYẾN NGHỊ SẢN PHẨM (RETAILROCKET DATASET)\n",
                    "## NOTEBOOK 02: THỬ NGHIỆM VÀ ĐÁNH GIÁ MÔ HÌNH LỌC CỘNG TÁC ALS\n",
                    "**Thành viên thực hiện:** HUY (Kỹ sư Học máy - Machine Learning Engineer)\n",
                    "\n",
                    "### Nội dung thực hiện:\n",
                    "1. Nạp ma trận tương tác User-Item từ tệp Parquet.\n",
                    "2. Khởi tạo phiên làm việc Apache Spark và phân chia Train / Test (80/20).\n",
                    "3. Xây dựng mô hình ALS với cơ chế Implicit Feedback (`implicitPrefs=True`).\n",
                    "4. Đánh giá sai số dự đoán qua thước đo RMSE và MAE.\n",
                    "5. Tinh chỉnh siêu tham số (Hyperparameter Tuning) và so sánh hiệu năng.\n",
                    "6. Trích xuất gợi ý sản phẩm cá nhân hóa Top-K."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pyspark.sql import SparkSession\n",
                    "from pyspark.ml.recommendation import ALS\n",
                    "from pyspark.ml.evaluation import RegressionEvaluator\n",
                    "from pyspark.sql.functions import col\n",
                    "\n",
                    "# Khởi tạo Spark Session\n",
                    "spark = SparkSession.builder \\\n",
                    "    .appName('Notebook_Test_ALS_Model') \\\n",
                    "    .config('spark.master', 'local[*]') \\\n",
                    "    .config('spark.driver.memory', '2g') \\\n",
                    "    .getOrCreate()\n",
                    "\n",
                    "spark.sparkContext.setLogLevel('WARN')\n",
                    "print('Spark Version:', spark.version)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Nạp ma trận tương tác User-Item 3 cột chuẩn"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "matrix_path = '../data/processed/user_item_matrix.parquet'\n",
                    "df_matrix = spark.read.parquet(matrix_path)\n",
                    "print(f'Tổng số cặp tương tác: {df_matrix.count():,}')\n",
                    "df_matrix.show(5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Phân chia tập dữ liệu Huấn luyện / Kiểm thử (80% Train, 20% Test)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_df, test_df = df_matrix.randomSplit([0.8, 0.2], seed=42)\n",
                    "train_df.cache()\n",
                    "test_df.cache()\n",
                    "print(f'Train records: {train_df.count():,}')\n",
                    "print(f'Test records: {test_df.count():,}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Huấn luyện mô hình ALS Implicit Feedback"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Cấu hình Best Model đã tìm thấy qua Grid Search\n",
                    "als = ALS(\n",
                    "    userCol='userId',\n",
                    "    itemCol='itemId',\n",
                    "    ratingCol='rating',\n",
                    "    implicitPrefs=True,\n",
                    "    coldStartStrategy='drop',\n",
                    "    rank=10,\n",
                    "    regParam=0.10,\n",
                    "    alpha=20.0,\n",
                    "    maxIter=10,\n",
                    "    seed=42\n",
                    ")\n",
                    "\n",
                    "print('Đang huấn luyện mô hình ALS...')\n",
                    "model = als.fit(train_df)\n",
                    "print('Huấn luyện hoàn tất!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Đánh giá mô hình trên tập kiểm thử (RMSE & MAE)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "predictions = model.transform(test_df)\n",
                    "\n",
                    "evaluator_rmse = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')\n",
                    "evaluator_mae = RegressionEvaluator(metricName='mae', labelCol='rating', predictionCol='prediction')\n",
                    "\n",
                    "rmse = evaluator_rmse.evaluate(predictions)\n",
                    "mae = evaluator_mae.evaluate(predictions)\n",
                    "\n",
                    "print(f'-> Đánh giá Test RMSE: {rmse:.4f}')\n",
                    "print(f'-> Đánh giá Test MAE:  {mae:.4f}')\n",
                    "predictions.show(5)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Sinh gợi ý Top-K sản phẩm cho người dùng mẫu"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample_users = test_df.select('userId').distinct().limit(5)\n",
                    "recommendations = model.recommendForUserSubset(sample_users, 5)\n",
                    "recommendations.show(truncate=False)"
                ]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open("E:/Bigdatadt3/Product_Recommender/notebooks/02_Test_ALS_Model.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
    print("-> Đã tạo thành công notebooks/02_Test_ALS_Model.ipynb")

if __name__ == "__main__":
    create_eda_notebook()
    create_als_notebook()
