# Carotid Ultrasound Arterial Stiffness Demo

本專案是一個以 **Streamlit** 建立的頸動脈超音波影像分析 Demo 系統。  
使用者可以透過網頁介面上傳 `.dcm` 格式的頸動脈超音波影像，手動選取血管 ROI 區域，系統會進行影像前處理、影像對位、血管輪廓追蹤，並估算血管直徑變化與 Beta arterial stiffness index。

> 本系統主要作為研究與展示用途，並非臨床診斷工具。

---

## 專案功能

- 上傳 DICOM 格式的頸動脈超音波影像
- 輸入病患收縮壓與舒張壓
- 透過網頁介面手動框選 ROI
- 對 DICOM 影像序列進行前處理
- 使用 ECC 進行影像對位
- 使用 Active Contour 追蹤血管輪廓
- 計算每一幀影像中的等效血管直徑
- 偵測直徑波形中的峰值與谷值
- 估算 Beta arterial stiffness index
- 顯示直徑變化波形圖與演算法驗證圖

---

## 系統畫面流程

系統主要分成三個步驟：

1. **輸入病患血壓值**  
   輸入收縮壓 SBP 與舒張壓 DBP，後續用於計算 Beta 指標。

2. **上傳影像與設定 ROI**  
   上傳 `.dcm` 檔案後，系統會顯示第一幀影像，使用者可直接拖曳紅框選取頸動脈區域。

3. **執行血管硬化分析**  
   系統會根據選取的 ROI 分析整段影像序列，並輸出：
   - 平均峰值直徑
   - 平均谷值直徑
   - Beta 硬化指標
   - 頸動脈直徑變化曲線
   - Active Contour 輪廓追蹤驗證圖

---

## 專案架構

```text
.
├── app.py        # Streamlit 網頁介面與結果顯示
├── logic.py      # DICOM 影像分析與血管直徑計算邏輯
└── README.md
