<h1 align="center">🎯 YOLO 即時人臉辨識系統</h1>

<p align="center">
  <strong>基於 YOLOv11 + MobileFaceNet 的生產級即時人臉辨識系統</strong><br/>
  增量學習 · 全方位角度覆蓋 · 全身場景支援
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ✨ 功能特色

| 功能 | 說明 |
|---|---|
| 🎯 **即時偵測** | YOLOv11-Pose 同時偵測人臉與姿態關鍵點 |
| 🧠 **增量學習** | 運行中自動收集多角度特徵向量（正面、側面、俯仰） |
| 🔄 **角度多樣性替換** | 自動淘汰冗餘向量，確保 360° 全方位覆蓋 |
| 📸 **品質篩選截圖** | 僅保留最清晰幀，自動排除模糊幀 |
| 🕴️ **全身場景支援** | 全身出現時仍可準確偵測並辨識人臉 |
| 🚫 **防誤判自動註冊** | 側臉等角度不會被誤認為陌生人並重複建檔 |
| ⚡ **多執行緒 Pipeline** | 鏡頭→偵測→辨識→資料庫，各自獨立執行緒 |
| 🗄️ **SQLite WAL** | 多執行緒並發讀寫，不鎖死 |
| 🖥️ **圖形介面** | 支援模型切換、閾值調整、警報設定 |

---

## 🚀 快速開始

### 系統需求

- Python **3.10+**
- USB 或內建攝影機
- RAM：4 GB+
- 作業系統：Windows 10+, macOS 12+, Ubuntu 20.04+, Raspberry Pi OS (64-bit)

### 安裝步驟

```bash
# 1. 建立虛擬環境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安裝依賴套件
pip install -r requirements.txt

# 3. 啟動程式
python main.py
```

首次執行時，程式將自動下載：
- `yolo11n-pose.pt`（約 6 MB）— YOLO 偵測模型
- `w600k_mbf.onnx`（約 16 MB）— MobileFaceNet 辨識模型

---

## 🎓 如何新增人物

### 透過攝影機（適合全身場景）
1. 點擊 **▶ 啟動辨識**，站在攝影機前方。
2. 約 3 秒後系統自動建立訪客檔案。
3. 在 **➕ 新增人物** 中為訪客重新命名。

### 透過照片（建議用於精準辨識）
1. 點擊 **➕ 新增人物** → 輸入姓名 → 選擇照片。
2. 建議提供 **8–15 張**，涵蓋以下角度：

| 角度 | 說明 |
|---|---|
| 正面 | 直視鏡頭，自然表情 |
| 左側 45° | 向左轉約 45 度 |
| 右側 45° | 向右轉約 45 度 |
| 左側 90° | 完全側面 |
| 右側 90° | 完全側面 |
| 微低頭 | 頭略微下傾 |
| 微抬頭 | 頭略微上仰 |
| 全身 | 距鏡頭 2–3 公尺，臉部清晰可見 |

---

## ⚙️ 設定參數

修改 `config.py` 調整效能：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | 相似度閾值（越低越嚴格） |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | 每人最多特徵向量數 |
| `SKIP_FRAMES` | 2 | 每 N 幀偵測一次（越大越省效能）|
| `DETECT_CONFIDENCE` | 0.45 | YOLO 偵測信心閾值 |
| `DETECT_IMG_SIZE` | 640 | 偵測解析度（Pi 建議 320） |
| `DETECT_DEVICE` | "cpu" | "cuda:0" 啟用 GPU |
| `CAMERA_INDEX` | 0 | 攝影機裝置編號 |

---

## 🏗️ 技術架構

```
CameraThread ──► DetectionThread ──► RecognitionThread ──► GUI 主執行緒
 (OpenCV)         (YOLOv11-Pose)     (MobileFaceNet)        (Tkinter)
                                            ↓
                                     DBWriterThread (SQLite WAL)
```

- **YOLOv11-Pose**：無 NMS 端對端推理，CPU 最佳化
- **MobileFaceNet**：512 維餘弦相似度辨識
- **增量學習**：每 1 秒觸發一次，自動追加多角度向量
- **角度多樣性防護**：相似度矩陣防止相同角度重複儲存

---

## ❓ 常見問題

**攝影機無法開啟？**
- 確認無其他程式占用攝影機
- 嘗試將 `config.py` 中的 `CAMERA_INDEX` 改為 `1` 或 `2`
- macOS：前往 系統設定 → 隱私與安全性 → 相機，授予 Python 存取權限

**模型下載失敗？**
- 確認網路連線正常
- 手動將 `yolo11n-pose.pt` 放入 `data/models/`
- 手動從 `buffalo_sc.zip` 解壓 `w600k_mbf.onnx` 並放入 `data/models/`

**辨識準確率不佳？**
- 提供更多不同角度的照片（建議 8–15 張）
- 清除現有資料後重新用多角度照片建檔
- 將 `RECOGNITION_THRESHOLD` 降低至 `0.55`

**FPS 過低？**
- 將 `SKIP_FRAMES` 增加至 `4`
- 將 `DETECT_IMG_SIZE` 降低至 `320`
- 降低攝影機解析度：`CAMERA_WIDTH=320, CAMERA_HEIGHT=240`

---

## 📄 授權

MIT License — 詳見 [LICENSE](../LICENSE)
