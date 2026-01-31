import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import io
import time
import cv2
import logic  # 確保資料夾內有 logic.py

# --- 設定網頁配置 ---
st.set_page_config(page_title="頸動脈超音波硬化觀察系統", layout="wide")

st.title("🩺 頸動脈超音波硬化觀察系統")
st.markdown("Demo")

# --- 快取讀取函數 ---
@st.cache_resource(show_spinner=False) 
def load_dicom_data(file_content):
    dcm_data = pydicom.dcmread(io.BytesIO(file_content))
    pixel_array = dcm_data.pixel_array
    return dcm_data, pixel_array

# ==========================================
# 步驟 1: 輸入生理參數
# ==========================================
st.markdown("#### 1️⃣ 第一步：輸入病患血壓值")
st.info("輸入收縮壓/舒張壓以計算 Beta 指標")
col_bp1, col_bp2, col_bp3 = st.columns([1, 1, 2])
with col_bp1:
    sbp = st.number_input("收縮壓 (SBP)", 90, 200, 120)
with col_bp2:
    dbp = st.number_input("舒張壓 (DBP)", 40, 120, 80)
st.markdown("---")

# ==========================================
# 步驟 2: 設定 ROI 與上傳
# ==========================================
st.markdown("#### 2️⃣ 第二步：上傳影像與設定 ROI")

# --- 側邊欄：進階設定 ---
with st.sidebar:
    st.header("🔧 進階演算法參數")
    st.info("請參考主畫面的預覽圖，調整下方的紅框範圍。")
    with st.expander("ROI 裁切範圍設定", expanded=True):
        row_range = st.slider("Col Range (Y軸高度)", 0, 800, (220, 400))
        col_range = st.slider("Row Range (X軸寬度)", 0, 1000, (480, 650))
        mm_per_pixel = st.number_input("mm/pixel 解析度", value=30/475, format="%.5f")

# 解包變數
ROW0, ROW1 = row_range
COL0, COL1 = col_range

uploaded_file = st.file_uploader("請選擇 .dcm 檔案", type=['dcm'])

# ==========================================
# 步驟 3: 預覽與執行分析
# ==========================================
if uploaded_file is not None:
    try:
        with st.spinner('讀取影像中...'):
            bytes_data = uploaded_file.getvalue()
            dcm_data, pixel_array = load_dicom_data(bytes_data)
            
            is_video = False
            if 'NumberOfFrames' in dcm_data and dcm_data.NumberOfFrames > 1: is_video = True
            elif len(pixel_array.shape) == 4 or (len(pixel_array.shape) == 3 and pixel_array.shape[0] > 10): is_video = True

        st.success(f"讀取成功！ID: {dcm_data.get('PatientID', 'Unknown')} | 尺寸: {pixel_array.shape}")

        # --- 預覽區塊 (含座標尺、紅框、中心點) ---
        if is_video:
            total_frames = pixel_array.shape[0]
            with st.expander("📂 點擊查看原始影像預覽 (含 ROI 中心點輔助)", expanded=True):
                
                c1, c2 = st.columns([1, 3])
                with c1:
                    preview_idx = st.number_input("預覽 Frame ID", 0, total_frames-1, 0)
                with c2:
                    st.write("")
                    st.caption("🟥 紅框：裁切範圍 | 🟡 黃點：ROI 中心 (請盡量對準血管中心)")

                # 準備繪圖
                original_img = pixel_array[preview_idx]
                
                # 建立畫布
                fig_preview, ax_preview = plt.subplots(figsize=(10, 6))
                
                if len(original_img.shape) == 3:
                    ax_preview.imshow(original_img)
                else:
                    ax_preview.imshow(original_img, cmap='gray')
                
                # 1. 畫 ROI 紅框
                rect_roi = patches.Rectangle(
                    (COL0, ROW0),
                    COL1 - COL0,
                    ROW1 - ROW0,
                    linewidth=2, 
                    edgecolor='red', 
                    facecolor='none', 
                    label='ROI Area'
                )
                ax_preview.add_patch(rect_roi)
                
                # 2. 畫 ROI 中心點 (修正請求 2)
                roi_center_y = (ROW0 + ROW1) / 2
                roi_center_x = (COL0 + COL1) / 2
                
                # 使用黃色十字加圓點，確保在深色背景清楚可見
                ax_preview.plot(roi_center_x, roi_center_y, marker='+', color='yellow', markersize=15, markeredgewidth=2, label='ROI Center')
                
                # 設定標題與軸
                ax_preview.set_title(f"Preview Frame {preview_idx}")
                ax_preview.set_xlabel("X Axis (Pixel)")
                ax_preview.set_ylabel("Y Axis (Pixel)")
                ax_preview.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
                ax_preview.legend(loc='upper right')
                
                st.pyplot(fig_preview)

        st.markdown("---")
        st.markdown("#### 3️⃣ 第三步：執行 血管硬化 分析")

        if st.button("🚀 開始完整分析", type="primary"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(percent, text):
                progress_bar.progress(percent)
                status_text.text(text)

            # 呼叫 logic.py
            results = logic.analyze_dicom_series(
                pixel_array=pixel_array,
                roi_coords=(ROW0, ROW1, COL0, COL1),
                mm_per_pixel=mm_per_pixel,
                progress_callback=update_progress
            )

            # 取回結果
            Ds_arr = results['diameter_array']
            peaks = results['peaks']
            valleys = results['valleys']
            aligned_imgs = results['aligned_imgs'] # 這是處理過的
            snake_list = results['snake_list']

            # 計算 Beta
            if len(peaks) > 0 and len(valleys) > 0:
                Ds = np.mean(Ds_arr[peaks])
                Dd = np.mean(Ds_arr[valleys])
                try:
                    strain = (Ds - Dd) / Dd
                    beta_val = np.log(sbp / dbp) / strain
                except: beta_val = 0
            else:
                Ds, Dd, beta_val = 0, 0, 0
                st.warning("訊號品質不佳，無法偵測波峰波谷")

            progress_bar.progress(100)
            status_text.success("分析完成！")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

            # 顯示報告
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("收縮壓", f"{sbp}")
            col2.metric("直徑平均峰值", f"{Ds:.2f} mm")
            col3.metric("直徑平均谷值", f"{Dd:.2f} mm")
            col4.metric("硬化指標", f"{beta_val:.2f}", delta_color="inverse")

            # 波形圖
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(Ds_arr, label='Diameter', color='#1f77b4')
            if len(peaks) > 0: ax.plot(peaks, Ds_arr[peaks], "x", color='red')
            if len(valleys) > 0: ax.plot(valleys, Ds_arr[valleys], "o", color='green')
            ax.set_title("Carotid Artery Diameter Change")
            ax.legend()
            st.pyplot(fig)

            st.markdown("---")

            # --- 驗證圖 (修正請求 1：調整大小與置中) ---
            st.markdown("#### 🟢 演算法驗證 (Frame 0)")
            st.caption("為確保分析正確，請確認紅線貼合血管內壁，且藍圓大小合理。")
            
            # 使用 columns 來限制圖片寬度
            # [1, 1, 1] 表示分成三等份，我們把圖放在中間那份
            c_left, c_center, c_right = st.columns([1, 1, 1])
            
            with c_center:
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                
                # 重新裁切原始影像作為底圖 (比較美觀)
                if len(pixel_array.shape) == 4:
                    raw_roi_0 = pixel_array[0, ROW0:ROW1, COL0:COL1, :]
                    raw_roi_0 = cv2.cvtColor(raw_roi_0, cv2.COLOR_RGB2GRAY)
                else:
                    raw_roi_0 = pixel_array[0, ROW0:ROW1, COL0:COL1]

                ax2.imshow(raw_roi_0, cmap="gray")
                
                # 畫 Snake
                snake_0 = snake_list[0]
                snake_closed = np.vstack([snake_0, snake_0[0]])
                ax2.plot(snake_closed[:, 1], snake_closed[:, 0], '-r', lw=1.5, alpha=0.8, label='Snake')

                # 畫等效圓
                d_mm = Ds_arr[0]
                d_pixel = d_mm / mm_per_pixel
                cy, cx = np.mean(snake_0[:, 0]), np.mean(snake_0[:, 1])
                circle = plt.Circle((cx, cy), d_pixel/2, color='cyan', fill=False, lw=2.5, label='Equiv. Circle')
                ax2.add_patch(circle)
                ax2.plot(cx, cy, '+', color='cyan')
                
                ax2.set_aspect('equal')
                ax2.axis('off')
                ax2.legend(loc='upper right', fontsize='small')
                
                # 顯示縮小後的圖片
                st.pyplot(fig2)

    except Exception as e:
        st.error(f"錯誤：{e}")

else:
    st.info("👈 請完成步驟 1 並在下方上傳檔案")