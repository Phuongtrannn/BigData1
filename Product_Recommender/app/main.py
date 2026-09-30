import streamlit as st
import recommender

# Thiết lập UI nâng cao
st.set_page_config(
    page_title="Hệ Khuyến Nghị Sản Phẩm", 
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# TUYỆT CHIÊU CSS: Biến Streamlit thành Web xịn như React
# ---------------------------------------------------------
custom_css = """
<style>
/* Ẩn các thành phần thừa của Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Đổi màu nền toàn trang để sang trọng hơn */
.stApp {
    background-color: #F7F7F9;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Hiệu ứng cho Card Sản phẩm (Container) */
div[data-testid="stVerticalBlock"] > div[style*="border"] {
    border: none !important;
    background: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03) !important;
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}
div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
    transform: translateY(-8px) !important;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08) !important;
}

/* Bo góc cho toàn bộ ảnh */
img {
    border-radius: 12px !important;
}

/* Styling cho Nút Bấm Chính (Primary) */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 20px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #333333 !important;
}

/* Styling cho Nút Bấm Phụ (Secondary - Nút Xem chi tiết) */
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: #E5E7EB !important;
    border-color: transparent !important;
    color: #000000 !important;
}

/* Tinh chỉnh typography tiêu đề */
h1 {
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    color: #111827 !important;
}
h3 {
    font-weight: 700 !important;
    color: #111827 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# SESSION STATE ĐIỀU HƯỚNG
# ---------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = 124

def go_to_home():
    st.session_state.page = 'home'
    st.session_state.selected_product = None

def go_to_detail(product_id, product_name):
    st.session_state.page = 'detail'
    st.session_state.selected_product = {"id": product_id, "name": product_name}

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x50.png?text=LOGO", use_container_width=True)
    st.markdown("### 👤 Đang đăng nhập")
    user_input = st.text_input("User ID của bạn:", str(st.session_state.current_user))
    
    if user_input.isdigit():
        st.session_state.current_user = int(user_input)
        
    st.info("💡 Mẹo: Nhập số chẵn để test AI, số lẻ để test Cold Start.")
    st.markdown("---")
    st.button("⬅️ Trở Về Cửa Hàng", on_click=go_to_home, use_container_width=True)

# =========================================================
# TRANG 1: TRANG CHỦ
# =========================================================
if st.session_state.page == 'home':
    st.markdown("<h1 style='text-align: center;'>BỘ SƯU TẬP MỚI NHẤT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280; margin-bottom: 40px;'>Khám phá những xu hướng thịnh hành nhất dành riêng cho bạn.</p>", unsafe_allow_html=True)
    
    popular_df = recommender.get_popular_products()
    cols = st.columns(4, gap="large")
    for idx, row in popular_df.iterrows():
        with cols[idx % 4]:
            with st.container(border=True):
                st.image(f"https://picsum.photos/400/500?random={row['item_id']}", use_container_width=True)
                st.markdown(f"<h4 style='margin: 10px 0 0 0;'>{row['name']}</h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: #F59E0B; font-weight: bold; margin: 0 0 10px 0;'>⭐ {row['score']}/5.0</p>", unsafe_allow_html=True)
                
                if st.button("Xem ngay", key=f"home_btn_{row['item_id']}", use_container_width=True):
                    go_to_detail(row['item_id'], row['name'])
                    st.rerun()

# =========================================================
# TRANG 2: CHI TIẾT SẢN PHẨM & AI RECOMMENDATION
# =========================================================
elif st.session_state.page == 'detail':
    prod = st.session_state.selected_product
    
    # Nút quay lại dạng text link (dùng markdown + css)
    st.button("← Tiếp tục mua sắm", on_click=go_to_home)
    st.write("")
    
    # Khu vực chi tiết sản phẩm chính
    col_img, col_info = st.columns([1, 1.2], gap="large")
    with col_img:
        st.image(f"https://picsum.photos/600/700?random={prod['id']}", use_container_width=True)
        
    with col_info:
        st.markdown(f"<h1 style='font-size: 3rem; margin-bottom: 0;'>{prod['name']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #6B7280; font-size: 1.2rem; margin-bottom: 20px;'>Mã Sản Phẩm: #{prod['id']}</p>", unsafe_allow_html=True)
        
        # Tags
        st.markdown("""
        <span style='background: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;'>✓ Còn hàng</span>
        <span style='background: #FEF3C7; color: #92400E; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; margin-left: 8px;'>⭐ 4.9 Đánh giá</span>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("Sản phẩm được thiết kế tỉ mỉ, tối ưu trải nghiệm người dùng với độ hoàn thiện cao cấp nhất. Một sự lựa chọn không thể thiếu trong giỏ hàng của bạn.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Thêm Vào Giỏ Hàng - $99.00", type="primary", use_container_width=True)
        
        with st.expander("Chính sách vận chuyển & Đổi trả", expanded=False):
            st.write("- Miễn phí giao hàng toàn quốc.\n- Hỗ trợ đổi trả trong 30 ngày.\n- Bảo hành chính hãng 1 năm.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ---------------------------------------------------------
        # HỆ KHUYẾN NGHỊ (DỜI LÊN ĐỂ LẤP KHOẢNG TRỐNG)
        # ---------------------------------------------------------
        st.markdown("#### ✨ Có thể bạn sẽ thích")
        
        user_id = st.session_state.current_user
        is_new = recommender.check_new_user(user_id)
        
        if is_new:
            st.caption(f"💡 Khách mới (User {user_id}): Top bán chạy.")
            rec_df = recommender.get_popular_products()
        else:
            st.caption(f"🎯 Gợi ý riêng cho User {user_id} từ AI (ALS).")
            rec_df = recommender.get_recommendations(user_id)
            
        # Do nằm trong cột nên ta chia 3 cột nhỏ gọn hơn
        r_cols = st.columns(3, gap="small")
        
        # Dùng counter để chỉ render tối đa 3 sản phẩm vào 3 cột
        col_idx = 0
        for idx, row in rec_df.iterrows():
            if row['item_id'] == prod['id']: 
                continue
            if col_idx >= 3:
                break
                
            with r_cols[col_idx]:
                with st.container(border=True):
                    # Ảnh vuông tỷ lệ 1:1 cho gọn
                    st.image(f"https://picsum.photos/300/300?random={row['item_id'] + 100}", use_container_width=True)
                    st.markdown(f"<p style='font-weight: 600; font-size: 0.9rem; margin: 5px 0 0 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{row['name']}</p>", unsafe_allow_html=True)
                    if st.button("Xem", key=f"rec_btn_{row['item_id']}", use_container_width=True):
                        go_to_detail(row['item_id'], row['name'])
                        st.rerun()
            col_idx += 1
