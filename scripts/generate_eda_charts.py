import os
import sys
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("=== [PHƯƠNG] BẮT ĐẦU PHÂN TÍCH DỮ LIỆU EDA & XUẤT BIỂU ĐỒ ===")
    
    raw_path = "E:/Bigdatadt3/Product_Recommender/data/raw/events.csv"
    fig_dir = "E:/Bigdatadt3/Product_Recommender/reports/figures/eda"
    metrics_path = "E:/Bigdatadt3/Product_Recommender/reports/eda_metrics.json"
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Đọc dữ liệu
    print(f"[1/6] Đang nạp tập dữ liệu: {raw_path}...")
    df = pd.read_csv(raw_path)
    total_records = len(df)
    print(f"-> Đã nạp thành công {total_records:,} bản ghi.")
    
    # 2. Xử lý thời gian và cấu trúc
    print("[2/6] Xử lý trường thời gian và trích xuất đặc trưng...")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_order_vn = ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'Chủ nhật']
    
    # 3. Tính toán các chỉ số thống kê (Metrics)
    print("[3/6] Thống kê chỉ số phân tích...")
    n_users = df['visitorid'].nunique()
    n_items = df['itemid'].nunique()
    event_counts = df['event'].value_counts()
    
    view_count = int(event_counts.get('view', 0))
    cart_count = int(event_counts.get('addtocart', 0))
    buy_count = int(event_counts.get('transaction', 0))
    
    view_pct = round(view_count / total_records * 100, 2)
    cart_pct = round(cart_count / total_records * 100, 2)
    buy_pct = round(buy_count / total_records * 100, 2)
    
    cart_per_view = round(cart_count / view_count * 100, 2)
    buy_per_cart = round(buy_count / cart_count * 100, 2)
    overall_conv = round(buy_count / view_count * 100, 2)
    
    user_interactions = df.groupby('visitorid').size()
    item_interactions = df.groupby('itemid').size()
    
    metrics = {
        "dataset_name": "RetailRocket Recommender System Dataset",
        "total_records": total_records,
        "unique_users": n_users,
        "unique_items": n_items,
        "min_date": str(df['datetime'].min()),
        "max_date": str(df['datetime'].max()),
        "event_counts": {
            "view": {"count": view_count, "percent": view_pct},
            "addtocart": {"count": cart_count, "percent": cart_pct},
            "transaction": {"count": buy_count, "percent": buy_pct}
        },
        "conversion_rates": {
            "view_to_cart_percent": cart_per_view,
            "cart_to_transaction_percent": buy_per_cart,
            "overall_conversion_percent": overall_conv
        },
        "user_activity": {
            "mean_interactions": round(float(user_interactions.mean()), 2),
            "median_interactions": float(user_interactions.median()),
            "max_interactions": int(user_interactions.max()),
            "pct_users_single_interaction": round(float((user_interactions == 1).mean() * 100), 2),
            "pct_users_ge_3_interactions": round(float((user_interactions >= 3).mean() * 100), 2)
        },
        "item_activity": {
            "mean_interactions": round(float(item_interactions.mean()), 2),
            "median_interactions": float(item_interactions.median()),
            "max_interactions": int(item_interactions.max())
        }
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
    print(f"-> Đã lưu metrics vào {metrics_path}")

    # Cấu hình phong cách biểu đồ trực quan chuẩn khoa học
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'Segoe UI', 'font.size': 11})

    # ==========================================================
    # BIỂU ĐỒ 1: PHÂN BỐ CÁC LOẠI HÀNH VI TƯƠNG TÁC (EVENT DISTRIBUTION)
    # ==========================================================
    print("[4/6] Đang vẽ Biểu đồ 1: Phân bố Event Type...")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#3B82F6', '#F59E0B', '#10B981']
    bars = ax.bar(['View (Xem)', 'AddToCart (Giỏ hàng)', 'Transaction (Mua)'], 
                  [view_count, cart_count, buy_count], color=colors, width=0.55, edgecolor='black', alpha=0.85)
    ax.set_yscale('log')
    ax.set_title("Phân bố số lượng các loại hành vi người dùng (Thang Logarit)", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Số lượng sự kiện (Log scale)", fontsize=12)
    ax.set_xlabel("Loại hành vi (Event Type)", fontsize=12)
    
    for bar, count, pct in zip(bars, [view_count, cart_count, buy_count], [view_pct, cart_pct, buy_pct]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, f"{count:,}\n({pct}%)", 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1F2937')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_event_distribution.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 2: PHÂN BỐ THEO KHUNG GIỜ HOẠT ĐỘNG (HOURLY ACTIVITY)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 2: Hoạt động theo giờ trong ngày...")
    hourly_df = df.groupby(['hour', 'event']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly_df.index, hourly_df['view'], marker='o', label=f"View (xem)", color='#3B82F6', linewidth=2.5)
    ax2 = ax.twinx()
    ax2.plot(hourly_df.index, hourly_df['addtocart'], marker='s', label=f"AddToCart", color='#F59E0B', linewidth=2, linestyle='--')
    ax2.plot(hourly_df.index, hourly_df['transaction'], marker='^', label=f"Transaction", color='#10B981', linewidth=2, linestyle='-.')
    
    ax.set_title("Phân bố hành vi người dùng theo từng khung giờ trong ngày (24 Giờ)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Khung giờ (UTC 0h - 23h)", fontsize=12)
    ax.set_ylabel("Số lượt Xem (View)", fontsize=12, color='#3B82F6')
    ax2.set_ylabel("Số lượt AddToCart & Transaction", fontsize=12, color='#10B981')
    ax.set_xticks(range(0, 24))
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_hourly_activity.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 3: PHÂN BỐ HOẠT ĐỘNG THEO THỨ TRONG TUẦN (DAY OF WEEK)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 3: Hoạt động theo thứ trong tuần...")
    dow_counts = df['day_of_week'].value_counts().reindex(day_order)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(day_order_vn, dow_counts.values, color='#6366F1', width=0.6, edgecolor='black', alpha=0.85)
    ax.set_title("Mật độ tương tác của người dùng theo các ngày trong tuần", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Tổng số lượt tương tác", fontsize=12)
    ax.set_ylim(0, dow_counts.max() * 1.15)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10000, f"{yval:,}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_day_of_week_activity.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 4: PHÂN BỐ TƯƠNG TÁC NGƯỜI DÙNG (USER INTERACTION LONG-TAIL)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 4: Phân bố tương tác người dùng (Long-Tail)...")
    bins = [1, 2, 3, 5, 10, 20, 50, 100, user_interactions.max() + 1]
    labels_bins = ['1', '2', '3-4', '5-9', '10-19', '20-49', '50-99', '>=100']
    user_bin_counts = pd.cut(user_interactions, bins=bins, right=False, labels=labels_bins).value_counts().reindex(labels_bins)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels_bins, user_bin_counts.values, color='#8B5CF6', edgecolor='black', width=0.6, alpha=0.85)
    ax.set_yscale('log')
    ax.set_title("Phân bố số lượng tương tác trên mỗi người dùng (Hiện tượng Long-Tail)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Số lượt tương tác / người dùng", fontsize=12)
    ax.set_ylabel("Số lượng người dùng (Log scale)", fontsize=12)
    for bar, count in zip(bars, user_bin_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2.0, count * 1.2, f"{count:,}\n({round(count/n_users*100, 1)}%)", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "04_user_interaction_longtail.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 5: TOP 10 SẢN PHẨM PHỔ BIẾN NHẤT (POPULAR ITEMS)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 5: Top 10 sản phẩm thịnh hành nhất...")
    top10_items = item_interactions.sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y_labels = [f"Item #{int(i)}" for i in top10_items.index]
    bars = ax.barh(y_labels[::-1], top10_items.values[::-1], color='#EC4899', edgecolor='black', height=0.6, alpha=0.85)
    ax.set_title("Top 10 sản phẩm có tổng lượt tương tác cao nhất (Cold-Start Candidates)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Tổng số lượt tương tác", fontsize=12)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 50, bar.get_y() + bar.get_height()/2.0, f" {int(w):,}", ha='left', va='center', fontsize=10, fontweight='bold')
    ax.set_xlim(0, top10_items.max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_top10_products.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 6: PHỄU CHUYỂN ĐỔI THƯƠNG MẠI ĐIỆN TỬ (CONVERSION FUNNEL)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 6: Phễu chuyển đổi E-commerce Funnel...")
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ['Xem hàng (View)', 'Thêm giỏ hàng (AddToCart)', 'Mua hàng (Transaction)']
    counts = [view_count, cart_count, buy_count]
    ax.barh(stages[::-1], counts[::-1], color=['#10B981', '#F59E0B', '#3B82F6'], height=0.55, edgecolor='black', alpha=0.9)
    ax.set_xscale('log')
    ax.set_title("Phễu chuyển đổi hành vi người dùng (E-Commerce Funnel)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Số lượt tương tác (Log scale)", fontsize=12)
    
    # Ghi chú tỷ lệ chuyển đổi
    ax.text(view_count * 0.1, 2, f"100% ({view_count:,})", color='white', fontweight='bold', va='center', fontsize=11)
    ax.text(cart_count * 0.05, 1, f"{cart_per_view}% so với View ({cart_count:,})", color='white', fontweight='bold', va='center', fontsize=10)
    ax.text(buy_count * 0.05, 0, f"{buy_per_cart}% so với Cart | {overall_conv}% so với View ({buy_count:,})", color='white', fontweight='bold', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "06_conversion_funnel.png"), dpi=300)
    plt.close()

    # ==========================================================
    # BIỂU ĐỒ 7: PHÂN BỐ ĐIỂM TƯƠNG TÁC QUY ĐỔI (RATING SCORE DISTRIBUTION)
    # ==========================================================
    print("      Đang vẽ Biểu đồ 7: Phân bố điểm Rating quy đổi...")
    # Tính điểm quy đổi: view = 1, addtocart = 3, transaction = 5
    # Thống kê điểm số tương tác tích lũy per (visitor, item)
    event_weights = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df['weight'] = df['event'].map(event_weights)
    user_item_scores = df.groupby(['visitorid', 'itemid'])['weight'].sum()
    
    score_bins = [1, 2, 3, 5, 8, 15, user_item_scores.max() + 1]
    score_labels = ['1 (View)', '2 (Multi-view)', '3-4 (Cart)', '5-7 (Buy)', '8-14 (Cart+Buy)', '>=15 (Repeat Buy)']
    score_dist = pd.cut(user_item_scores, bins=score_bins, right=False, labels=score_labels).value_counts().reindex(score_labels)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(score_labels, score_dist.values, color='#0D9488', edgecolor='black', width=0.6, alpha=0.85)
    ax.set_yscale('log')
    ax.set_title("Phân bố điểm tương tác quy đổi Implicit Rating theo cặp (User, Item)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Mức điểm tương tác quy đổi", fontsize=12)
    ax.set_ylabel("Số cặp User-Item (Log scale)", fontsize=12)
    for bar, count in zip(bars, score_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2.0, count * 1.2, f"{count:,}\n({round(count/len(user_item_scores)*100, 1)}%)", 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "07_rating_score_distribution.png"), dpi=300)
    plt.close()

    print("[6/6] === XUẤT 7 BIỂU ĐỒ EDA THÀNH CÔNG VÀO reports/figures/eda/! ===")

if __name__ == "__main__":
    main()
