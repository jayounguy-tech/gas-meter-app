import streamlit as st
import streamlit.components.v1 as components  # 引入元件庫，用於執行 JavaScript
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
    /* 調整相機輸入框樣式 */
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
# 4. 核心辨識邏輯
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
    current_conf = 0.4
    min_conf = 0.1
    step = 0.05
    imgsz_setting = 1280
    
    final_res_image = None
    final_reading = ""
    final_serial = ""
    used_conf = current_conf

    while current_conf >= min_conf:
        results = model(image_input, conf=current_conf, iou=0.5, imgsz=imgsz_setting, verbose=False)
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
# 5. UI 介面設計
# ==========================================

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
    
    # 顯示相機
    camera_file = st.camera_input("請對準瓦斯表拍攝")
    if camera_file:
        image_source = Image.open(camera_file)
else:
    uploaded_file = st.file_uploader("選擇照片", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image_source = Image.open(uploaded_file)

# ==========================================
# 6. 執行辨識
# ==========================================
if image_source is not None:
    with st.spinner('🤖 AI 正在嘗試最佳參數辨識中...'):
        processed_img, reading_str, serial_str, final_conf = process_image_adaptive(image_source)
    
    st.markdown("### 📊 辨識結果")
    
    if final_conf < 0.4:
        st.caption(f"ℹ️ 已自動降低信心度至 **{final_conf}** 以獲取更多數字")

    col1, col2 = st.columns(2)
    with col1:
        if len(reading_str) >= 4:
            st.metric("🔥 度數", reading_str)
        else:
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
```

### ✨ 更新重點：
1.  **新增 `import streamlit.components.v1 as components`：** 這是用來執行 JavaScript 的模組。
2.  **新增 `inject_torch_control` 函式：**
    * 這段程式碼會在背景偷偷執行 JavaScript。
    * 它會去尋找瀏覽器中的 `<video>` 標籤（也就是相機畫面）。
    * 如果找到，它會嘗試設定 `torch: true`（開啟手電筒）。
    * 如果不支援（例如 iPhone 或是電腦 Webcam），它會在 Console 報錯但不會讓網頁當機。
3.  **介面新增 Toggle 開關：**
    * 在相機模式上方多了一個 `🔦 開啟補光燈 (Android)` 的開關。
    * **注意：** 這個開關切換時，網頁會重新整理是正常的 Streamlit 行為。

### 🚀 如何更新伺服器？
1.  將這份新程式碼存成 `app.py`。
2.  **Commit & Push** 到 GitHub。
3.  **Streamlit Cloud** 會自動偵測到更新並重新部署。

快去用 Android 手機試試看吧！(iPhone 如果沒反應是正常的系統限制喔)。