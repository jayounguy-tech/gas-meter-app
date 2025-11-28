import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ==========================================
# 1. 頁面基礎設定
# ==========================================
st.set_page_config(
    page_title="瓦斯表 AI 辨識",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #ff4b4b;
    }
    .stCameraInput {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 載入模型
# ==========================================
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"找不到模型檔案 best.pt，請確認檔案位置！\n錯誤: {e}")
    st.stop()

st.title("🔥 瓦斯表抄表助手")

# ==========================================
# 3. 核心邏輯 (含自適應迴圈)
# ==========================================

def is_inside(cx, cy, box_obj):
    if box_obj is None: return False
    bx1, by1, bx2, by2 = box_obj['coords']
    margin = 10
    in_box = (bx1 - margin < cx < bx2 + margin) and (by1 - margin < cy < by2 + margin)
    if not in_box: return False
    box_height = by2 - by1
    relative_y = (cy - by1) / box_height
    return 0.2 < relative_y < 0.8

def process_image_adaptive(image_input):
    """
    自適應處理函式：
    從信心度 0.4 開始嘗試，
    如果 度數 < 4碼 或 表號 < 6碼，就降低信心度重試。
    """
    
    # 初始設定
    current_conf = 0.4   # 起始信心度
    min_conf = 0.1       # 最低底限 (避免降到 0 抓到一堆雜訊)
    step = 0.1           # 每次降低多少 (10%)
    imgsz_setting = 1280 # 固定高解析度
    
    final_res_image = None
    final_reading = ""
    final_serial = ""
    used_conf = current_conf

    # --- 自適應迴圈 (Adaptive Loop) ---
    while current_conf >= min_conf:
        
        # 1. 執行預測
        results = model(image_input, conf=current_conf, iou=0.5, imgsz=imgsz_setting, verbose=False)
        result = results[0]
        img_h, img_w = result.orig_shape
        
        # 2. 解析資料
        gas_meter_box = None      
        serial_number_box = None  
        digits_found = []         

        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            if class_name == 'GasMeter':
                if gas_meter_box is None or conf > gas_meter_box['conf']:
                    gas_meter_box = {'coords': [x1, y1, x2, y2], 'conf': conf}
                    
            elif class_name == 'SerialNumber':
                # 表號擴大範圍 (Padding)
                pad_w, pad_h = 30, 10
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(img_w, x2 + pad_w)
                y2 = min(img_h, y2 + pad_h)
                
                if serial_number_box is None or conf > serial_number_box['conf']:
                    serial_number_box = {'coords': [x1, y1, x2, y2], 'conf': conf}
            
            elif class_name.isdigit():
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                digits_found.append({'val': class_name, 'cx': center_x, 'cy': center_y, 'x1': x1})

        # 3. 分配數字
        reading_digits = []
        serial_digits = []
        for d in digits_found:
            if is_inside(d['cx'], d['cy'], gas_meter_box):
                reading_digits.append(d)
            elif is_inside(d['cx'], d['cy'], serial_number_box):
                serial_digits.append(d)

        reading_digits.sort(key=lambda x: x['x1'])
        serial_digits.sort(key=lambda x: x['x1'])
        
        temp_reading = "".join([d['val'] for d in reading_digits])
        temp_serial = "".join([d['val'] for d in serial_digits])
        
        # 4. 檢查條件：是否滿足位數要求？
        # 條件：度數 >= 4碼 且 表號 >= 6碼 (表號有時候可能只有 5 或 8，可視情況調整)
        condition_met = (len(temp_reading) >= 4) and (len(temp_serial) >= 6)
        
        # 暫存這次的結果
        final_reading = temp_reading
        final_serial = temp_serial
        used_conf = current_conf
        
        # 產出圖片
        res_plotted = result.plot()
        final_res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # 5. 判斷是否要跳出迴圈
        if condition_met:
            break  # 成功抓齊了，收工！
        
        # 如果還沒抓齊，降低信心度，準備跑下一輪
        current_conf -= step
        
        # 防止浮點數運算誤差導致無限迴圈
        current_conf = round(current_conf, 2)

    return final_res_image, final_reading, final_serial, used_conf

# ==========================================
# 4. 手機版介面設計
# ==========================================
# 將設定隱藏在摺疊選單中，保持介面乾淨
with st.expander("⚙️ 辨識設定 (覺得不準請點這)", expanded=False):
    conf_thres = st.slider("信心度 (Confidence)", 0.1, 0.8, current_conf, 0.05)
    img_size = st.selectbox("解析度 (Img Size)", [640, 960, 1280], index=2)

# 圖片來源選擇
mode = st.radio("選擇輸入方式：", ["📸 開啟相機", "📤 上傳照片"], horizontal=True)

image_source = None

if mode == "📸 開啟相機":
    camera_file = st.camera_input("請對準瓦斯表拍攝")
    if camera_file:
        image_source = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader("選擇照片", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_source = Image.open(uploaded_file)

# ==========================================
# 5. 執行與顯示
# ==========================================
if image_source is not None:
    # 顯示載入動畫
    with st.spinner('🤖 AI 正在嘗試最佳參數辨識中...'):
        processed_img, reading_str, serial_str, final_conf = process_image_adaptive(image_source)
    
    st.markdown("### 📊 辨識結果")
    
    # 顯示最終使用的信心度 (讓你知道 AI 多努力)
    if final_conf < 0.4:
        st.caption(f"ℹ️ 已自動降低信心度至 **{final_conf}** 以獲取更多數字")

    col1, col2 = st.columns(2)
    with col1:
        if len(reading_str) >= 4:
            st.metric("🔥 度數", reading_str)
        else:
            # 如果降到最低還是抓不到，顯示紅色警告
            st.metric("🔥 度數", reading_str if reading_str else "N/A", delta="位數不足" if reading_str else "未偵測", delta_color="inverse")
            
    with col2:
        if len(serial_str) >= 6:
            st.metric("🔢 表號", serial_str)
        else:
             st.metric("🔢 表號", serial_str if serial_str else "N/A", delta="位數不足" if serial_str else "未偵測", delta_color="inverse")
            
    st.divider()

    img_tab1, img_tab2 = st.tabs(["👁️ 辨識結果圖", "📷 原始圖片"])
    
    with img_tab1:
        st.image(processed_img, caption=f"AI 繪製框線 (Conf: {final_conf})", use_container_width=True)
    with img_tab2:
        st.image(image_source, caption="原始上傳", use_container_width=True)


