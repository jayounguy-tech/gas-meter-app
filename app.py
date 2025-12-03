import streamlit as st
import streamlit.components.v1 as components  # 引入元件庫，用於執行 JavaScript
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
import gdown  # 記得在 requirements.txt 加入 gdown

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
    /* 調整相機輸入框樣式 */
    .stCameraInput {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 自動下載模型 (解決 GitHub 檔案限制)
# ==========================================
@st.cache_resource
def load_model():
    model_path = 'best.pt'
    
    # 檢查模型是否存在，不存在就下載
    if not os.path.exists(model_path):
        st.info("☁️ 正在從 Google Drive 下載模型 (約 40MB)，初次啟動需時較長，請稍候...")
        try:
            # ---------------------------------------------------------
            # ⚠️ 請將下方的 ID 換成你 Google Drive 檔案的 ID ⚠️
            # 網址範例: https://drive.google.com/file/d/1ABCDE.../view
            # ID 就是: 1ABCDE...
            # ---------------------------------------------------------
            file_id = '1-Wq7P73qno7w8sXWSKiC6lW4JG6uafpJ' 
            
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            st.success("✅ 下載完成！")
        except Exception as e:
            st.error(f"❌ 模型下載失敗！請檢查 Google Drive 權限是否設為公開，或 ID 是否正確。\n錯誤訊息: {e}")
            st.stop()
            
    return YOLO(model_path)

# 嘗試載入模型
try:
    model = load_model()
except Exception as e:
    st.error(f"模型載入發生錯誤: {e}")
    st.stop()

st.title("🔥 瓦斯表抄表助手")

# ==========================================
# 3. Javascript 補光燈控制邏輯
# ==========================================
def inject_torch_control(enable_torch):
    """
    注入 JavaScript 來控制瀏覽器的 MediaStream (補光燈)
    """
    torch_state = "true" if enable_torch else "false"
    
    js_code = f"""
    <script>
    // 設定計時器，因為相機可能還沒完全啟動，每 500ms 檢查一次
    var attempts = 0;
    var torchInterval = setInterval(function() {{
        // 嘗試抓取 Streamlit 的 video 標籤 (位於 iframe 父層)
        var video = window.parent.document.querySelector('video');
        
        if (video && video.srcObject) {{
            var track = video.srcObject.getVideoTracks()[0];
            
            // 檢查瀏覽器是否支援 image-capture (補光燈)
            var capabilities = track.getCapabilities();
            if (capabilities.torch) {{
                track.applyConstraints({{
                    advanced: [{{ torch: {torch_state} }}]
                }}).then(() => {{
                    console.log("補光燈狀態已切換為: {torch_state}");
                }}).catch(err => {{
                    console.log("補光燈切換失敗: ", err);
                }});
                
                // 成功抓到後，清除計時器
                clearInterval(torchInterval);
            }}
        }}
        
        attempts++;
        // 嘗試 10 次 (5秒) 後放棄，避免無限執行
        if (attempts > 10) clearInterval(torchInterval);
        
    }}, 500);
    </script>
    """
    # 注入 HTML/JS (高度設為 0 隱藏起來)
    components.html(js_code, height=0)


# ==========================================
# 4. 核心辨識邏輯 (含 Padding、自適應、防重疊)
# ==========================================

def is_inside(cx, cy, box_obj):
    """判斷數字中心點是否在大框內"""
    if box_obj is None: return False
    bx1, by1, bx2, by2 = box_obj['coords']
    margin = 10
    in_box = (bx1 - margin < cx < bx2 + margin) and (by1 - margin < cy < by2 + margin)
    if not in_box: return False
    
    # 垂直過濾：數字應該在框框高度的中間 20%~80% 區域
    box_height = by2 - by1
    relative_y = (cy - by1) / box_height
    return 0.2 < relative_y < 0.8

def remove_overlapping_digits(digits_list, iou_threshold=0.3):
    """
    移除重疊的數字框 (保留信心度高的)
    針對瓦斯表數字，我們特別關注 X 軸的重疊
    """
    if not digits_list:
        return []
    
    # 1. 依照信心度由高到低排序 (優先保留高信心的)
    sorted_digits = sorted(digits_list, key=lambda x: x['conf'], reverse=True)
    final_digits = []
    
    for current in sorted_digits:
        is_duplicate = False
        for kept in final_digits:
            # 計算 X 軸重疊 (1D IoU)
            # 兩個區間 [x1, x2] 的重疊長度
            x_left = max(current['x1'], kept['x1'])
            x_right = min(current['x2'], kept['x2'])
            overlap_width = max(0, x_right - x_left)
            
            # 計算較小那個框的寬度
            min_width = min(current['x2'] - current['x1'], kept['x2'] - kept['x1'])
            
            # 如果重疊超過寬度的 30%，視為重複 (或者是包含關係)
            if min_width > 0 and (overlap_width / min_width) > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_digits.append(current)
            
    return final_digits

def process_image_adaptive(image_input):
    current_conf = 0.4
    min_conf = 0.1
    step = 0.1
    imgsz_setting = 1280
    
    final_res_image = None
    final_reading = ""
    final_serial = ""
    used_conf = current_conf

    while current_conf >= min_conf:
        # 1. 執行預測
        # 【關鍵修改】加入 agnostic_nms=True，強制跨類別抑制重疊 (例如 3 和 8 重疊只留一個)
        results = model(image_input, conf=current_conf, iou=0.5, imgsz=imgsz_setting, agnostic_nms=True, verbose=False)
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
                # Padding 擴大
                pad_w, pad_h = 10, 10
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(img_w, x2 + pad_w)
                y2 = min(img_h, y2 + pad_h)
                
                if serial_number_box is None or conf > serial_number_box['conf']:
                    serial_number_box = {'coords': [x1, y1, x2, y2], 'conf': conf}
            
            elif class_name.isdigit():
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # 儲存更多資訊以便後續過濾 (x1, x2)
                digits_found.append({
                    'val': class_name, 
                    'cx': center_x, 
                    'cy': center_y, 
                    'x1': x1, 
                    'x2': x2, 
                    'conf': conf
                })

        # 2. 初步分類數字
        raw_reading_digits = []
        raw_serial_digits = []
        
        for d in digits_found:
            if is_inside(d['cx'], d['cy'], gas_meter_box):
                raw_reading_digits.append(d)
            elif is_inside(d['cx'], d['cy'], serial_number_box):
                raw_serial_digits.append(d)

        # 3. 【關鍵修改】執行防重疊過濾 (移除幽靈數字)
        reading_digits = remove_overlapping_digits(raw_reading_digits, iou_threshold=0.3)
        serial_digits = remove_overlapping_digits(raw_serial_digits, iou_threshold=0.3)

        # 4. 排序與組合
        reading_digits.sort(key=lambda x: x['x1'])
        serial_digits.sort(key=lambda x: x['x1'])
        
        temp_reading = "".join([d['val'] for d in reading_digits])
        temp_serial = "".join([d['val'] for d in serial_digits])
        
        condition_met = (len(temp_reading) >= 4) and (len(temp_serial) >= 6)
        
        final_reading = temp_reading
        final_serial = temp_serial
        used_conf = current_conf
        
        res_plotted = result.plot()
        final_res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        if condition_met:
            break
        
        current_conf -= step
        current_conf = round(current_conf, 2)

    return final_res_image, final_reading, final_serial, used_conf

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
    # -----------------------------------------------------
    # 🔦 補光燈開關 (僅在相機模式顯示)
    # -----------------------------------------------------
    col_t1, col_t2 = st.columns([0.4, 0.6])
    with col_t1:
        use_torch = st.toggle("🔦 開啟補光燈 (Android)", value=False)
        if use_torch:
            st.caption("嘗試開啟閃光燈...")
    
    # 注入 JS 控制碼
    inject_torch_control(use_torch)
    
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

