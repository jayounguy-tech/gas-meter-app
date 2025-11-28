import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ==========================================
# 1. 頁面基礎設定 (設定手機版面)
# ==========================================
st.set_page_config(
    page_title="瓦斯表 AI 辨識",
    page_icon="🔥",
    layout="centered",  # 手機版建議置中，不要 wide
    initial_sidebar_state="collapsed"
)

# 自訂 CSS 讓手機版更好看 (加大字體、隱藏選單)
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
    # 確保 best.pt 在同目錄下
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"找不到模型檔案 best.pt，請確認檔案位置！\n錯誤: {e}")
    st.stop()

st.title("🔥 瓦斯表抄表助手")

# ==========================================
# 3. 核心邏輯 (與之前相同)
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

def process_image(image_input, conf_thres, img_size):
    # 執行 YOLO 預測
    results = model(image_input, conf=conf_thres, iou=0.5, imgsz=img_size)
    result = results[0]
    img_h, img_w = result.orig_shape
    
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

    reading_digits = []
    serial_digits = []
    for d in digits_found:
        if is_inside(d['cx'], d['cy'], gas_meter_box):
            reading_digits.append(d)
        elif is_inside(d['cx'], d['cy'], serial_number_box):
            serial_digits.append(d)

    reading_digits.sort(key=lambda x: x['x1'])
    serial_digits.sort(key=lambda x: x['x1'])
    
    final_reading = "".join([d['val'] for d in reading_digits])
    final_serial = "".join([d['val'] for d in serial_digits])
    
    res_plotted = result.plot()
    res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return res_image, final_reading, final_serial

# ==========================================
# 4. 手機版介面設計
# ==========================================

# 將設定隱藏在摺疊選單中，保持介面乾淨
with st.expander("⚙️ 辨識設定 (覺得不準請點這)", expanded=False):
    conf_thres = st.slider("信心度 (Confidence)", 0.1, 0.8, 0.25, 0.05)
    img_size = st.selectbox("解析度 (Img Size)", [640, 960, 1280], index=2)

# 圖片來源選擇
mode = st.radio("選擇輸入方式：", ["📸 開啟相機", "📤 上傳照片"], horizontal=True)

image_source = None

if mode == "📸 開啟相機":
    # Streamlit 的相機功能在手機上非常好用
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
    with st.spinner('🤖 AI 正在用力辨識中...'):
        processed_img, reading_str, serial_str = process_image(image_source, conf_thres, img_size)
    
    # 手機版面：重點結果放最上面，且字體放大
    st.markdown("### 📊 辨識結果")
    
    col1, col2 = st.columns(2)
    with col1:
        if reading_str:
            st.metric("🔥 度數", reading_str)
        else:
            st.warning("度數未偵測")
            
    with col2:
        if serial_str:
            st.metric("🔢 表號", serial_str)
        else:
            st.warning("表號未偵測")
            
    st.divider()

    # 使用分頁切換圖片，節省垂直空間
    img_tab1, img_tab2 = st.tabs(["👁️ 辨識結果圖", "📷 原始圖片"])
    
    with img_tab1:
        st.image(processed_img, caption="AI 繪製框線", use_container_width=True)
    with img_tab2:
        st.image(image_source, caption="原始上傳", use_container_width=True)