# 檔案名稱: logic.py
import numpy as np
import cv2
from skimage.segmentation import active_contour
from scipy.signal import find_peaks

def analyze_dicom_series(pixel_array, roi_coords, mm_per_pixel, progress_callback=None):
    """
    這是一個封裝好的函式，負責執行所有影像處理邏輯。
    :param pixel_array: DICOM 影像數據 (T, H, W) 或 (T, H, W, 3)
    :param roi_coords: Tuple (row_start, row_end, col_start, col_end)
    :param mm_per_pixel: 解析度參數
    :param progress_callback: 用來更新 Streamlit 進度條的函式
    :return: 字典，包含計算結果與繪圖所需的資料
    """
    
    ROW0, ROW1, COL0, COL1 = roi_coords
    N_FRAMES = pixel_array.shape[0]
    
    # 1. 裁切與轉灰階
    cropped_imgs = []
    for i in range(N_FRAMES):
        if len(pixel_array.shape) == 4:
            cropped = pixel_array[i, ROW0:ROW1, COL0:COL1, :]
            gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        else:
            gray = pixel_array[i, ROW0:ROW1, COL0:COL1]
        cropped_imgs.append(gray)
        
        if progress_callback and i % 50 == 0:
            progress_callback(int(i / N_FRAMES * 10), "裁切 ROI 與 灰階轉換...")
            
    cropped_imgs = np.array(cropped_imgs)

    # 2. 前處理
    proc_imgs = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(N_FRAMES):
        gauss_img = cv2.GaussianBlur(cropped_imgs[i], (5, 5), 0)
        clahe_img = clahe.apply(gauss_img)
        proc_imgs.append(clahe_img)
        
        if progress_callback and i % 50 == 0:
            progress_callback(10 + int(i / N_FRAMES * 10), "影像增強 (Gaussian + CLAHE)...")
            
    proc_imgs = np.array(proc_imgs)

    # 3. ECC 對位
    ref = proc_imgs[0].astype(np.float32) / 255.0
    h_roi, w_roi = ref.shape
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix_init = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
    aligned_imgs = []

    for i in range(N_FRAMES):
        this_img = proc_imgs[i].astype(np.float32) / 255.0
        warp_matrix = warp_matrix_init.copy()
        try:
            cc, warp_matrix = cv2.findTransformECC(ref, this_img, warp_matrix, warp_mode, criteria)
            aligned = cv2.warpAffine(this_img, warp_matrix, (w_roi, h_roi), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            aligned = this_img
        aligned_imgs.append((aligned * 255).astype(np.uint8))
        
        if progress_callback and i % 10 == 0:
            progress_callback(20 + int(i / N_FRAMES * 40), "影像對位 (ECC Alignment)...")
            
    aligned_imgs = np.array(aligned_imgs)

    # 4. Active Contour
    snake_list = []
    
    # --- 修改開始：動態計算中心點 ---
    # 取得裁切後影像的長寬 (aligned_imgs 已經是裁切好的了)
    h_crop, w_crop = aligned_imgs[0].shape 
    
    # 自動設定圓心在「裁切圖」的正中間
    cx_snake = w_crop // 2
    cy_snake = h_crop // 2
    
    # 設定半徑 (可以是固定值，或是根據圖片大小動態調整，這裡先維持固定 50)
    # 如果 ROI 拉太小，半徑 50 可能會爆出去，可以加個保護機制：
    r_snake = min(h_crop, w_crop) // 4  # 例如：設為長寬最小邊的 1/4，這樣更保險
    
    t = np.linspace(0, 2 * np.pi, 100)
    init_snake = np.array([cy_snake + r_snake * np.sin(t), cx_snake + r_snake * np.cos(t)]).T

    for i in range(N_FRAMES):
        ret, bi_img = cv2.threshold(aligned_imgs[i], 105, 255, cv2.THRESH_BINARY)
        opening = cv2.morphologyEx(bi_img, cv2.MORPH_OPEN, (5, 5))
        snake = active_contour(opening, init_snake, alpha=0.1, beta=0.1, gamma=0.01, w_line=0, w_edge=1)
        snake_list.append(snake)
        
        if progress_callback and i % 20 == 0:
            progress_callback(60 + int(i / N_FRAMES * 30), "輪廓追蹤 (Active Contour)...")
            
    snake_list = np.array(snake_list, dtype=object)

    # 5. 計算直徑
    diameter_list = []
    for i in range(N_FRAMES):
        snake_i = snake_list[i]
        cnt = np.stack([snake_i[:, 1], snake_i[:, 0]], axis=1).astype(np.float32)
        area = cv2.contourArea(cnt)
        D_eq_pixel = 2 * np.sqrt(area / np.pi)
        D_eq_mm = D_eq_pixel * mm_per_pixel
        diameter_list.append(D_eq_mm)

    diameter_array = np.array(diameter_list)
    
    # 找峰值
    peaks, _ = find_peaks(diameter_array, prominence=0.08)
    valleys, _ = find_peaks(-diameter_array, prominence=0.08)

    return {
        "diameter_array": diameter_array,
        "peaks": peaks,
        "valleys": valleys,
        "aligned_imgs": aligned_imgs,
        "snake_list": snake_list
    }