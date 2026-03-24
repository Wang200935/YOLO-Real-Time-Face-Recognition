# YOLOv11 即時人臉辨識系統

即時人臉辨識系統，具備增量學習、批量導入、警報功能與圖形介面。

## 系統需求

- Python 3.8 – 3.12
- 攝影機（USB 或內建）
- RAM 建議 4 GB+
- 作業系統：Windows 10+, macOS 12+, Ubuntu 20.04+, Raspberry Pi OS (64-bit)

## 快速安裝

```bash
# 1. 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動程式
python main.py
```

首次執行時，程式會自動下載：
- **YOLOv11 nano 模型** (`yolo11n-pose.pt`) — 約 6 MB
- **MobileFaceNet ONNX** (`w600k_mbf.onnx`) — 約 16 MB（從 buffalo_sc.zip 提取）

## 使用說明

### 基本操作

1. **啟動辨識**：點擊「▶ 啟動辨識」按鈕，開啟攝影機即時偵測
2. **新增人物**：點擊「➕ 新增人物」，輸入姓名並上傳 1 張以上照片
3. **批量導入**：點擊「📁 批量導入」，選擇包含子資料夾的目錄
   - 結構：`資料夾/姓名A/img1.jpg`, `資料夾/姓名B/img1.jpg` …
   - 無子資料夾時，圖片自動命名為「訪客1」、「訪客2」…
4. **設定警報**：在人物列表右鍵選「🔔 切換警示」，偵測到該人物時發出警報
5. **重新命名**：雙擊列表中的名稱，或右鍵選「✏️ 重新命名」
6. **刪除人物**：右鍵選「🗑 刪除人物」（此操作無法還原）

### 效能調整

在 `config.py` 中修改：

| 參數 | 預設值 | 說明 |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.40 | 相似度閾值，降低可減少誤判 |
| `SKIP_FRAMES` | 2 | 每 N 幀執行一次檢測 |
| `DETECT_IMG_SIZE` | 640 | 檢測解析度（320 適合 Pi）|
| `DETECT_DEVICE` | "cpu" | "cuda:0" 啟用 GPU |
| `CAMERA_INDEX` | 0 | 多攝影機時修改此值 |

## 平台特定說明

### macOS
- 首次使用需在「系統設定 → 隱私與安全性 → 相機」允許 Python 存取攝影機
- Apple Silicon (M1/M2) 自動啟用 CoreML 加速

### Windows
- 攝影機使用 DirectShow 後端，若黑畫面可嘗試修改 `CAMERA_INDEX = 1`
- Tkinter 已內建於 python.org 官方安裝包

### Linux
```bash
sudo apt install python3-tk libsdl2-mixer-2.0-0
```

### Raspberry Pi
- 建議 Pi 4 或更新版本
- 自動降低解析度至 320x320，幀率至 15 FPS
- 使用 PiCamera 3 需額外安裝 `picamera2`：
  ```bash
  pip install picamera2
  ```

## 檔案結構

```
YOLOv11 級時人臉辨識系統/
├── main.py              # 程式入口
├── config.py            # 設定檔
├── face_detector.py     # YOLOv11 人臉檢測
├── face_recognizer.py   # MobileFaceNet 特徵提取 + 辨識
├── database.py          # SQLite 資料庫
├── gui.py               # Tkinter 圖形介面
├── worker_threads.py    # 多執行緒 Pipeline
├── utils.py             # 工具函式
├── requirements.txt
├── README.md
├── RISK_REVIEW.md
└── data/
    ├── models/          # AI 模型（首次執行自動下載）
    ├── db/              # faces.db
    ├── screenshots/     # 人物截圖
    └── logs/            # 執行日誌
```

## 常見問題

**Q: 攝影機無法開啟**
- 確認其他程式未佔用攝影機
- 嘗試修改 `config.py` 中的 `CAMERA_INDEX = 1`（或 2）
- macOS：確認已授權攝影機存取

**Q: 模型下載失敗**
- 確認網路連線
- 手動下載 `yolo11n-pose.pt` 放至 `data/models/`
- 手動下載 `buffalo_sc.zip` 並解壓取出 `w600k_mbf.onnx` 放至 `data/models/`

**Q: 辨識準確率不佳**
- 增加每個人物的訓練照片數量（建議 5 張以上，不同角度）
- 調低 `RECOGNITION_THRESHOLD`（如 0.35）
- 確保訓練照片清晰且人臉佔比大

**Q: FPS 太低**
- 增加 `SKIP_FRAMES`（如 4）
- 降低 `DETECT_IMG_SIZE`（如 320）
- 減少攝影機解析度：`CAMERA_WIDTH=320, CAMERA_HEIGHT=240`

## 技術架構

```
CameraThread ──► DetectionThread ──► RecognitionThread ──► GUI 主執行緒
  (CV2 擷取)    (YOLOv11 推理)       (MobileFaceNet)       (Tkinter)
                                           ↓
                                    DBWriterThread (SQLite WAL)
```

- **YOLOv11**：無 NMS 端對端推理，CPU 加速最佳化
- **MobileFaceNet**：512 維特徵向量，餘弦相似度辨識
- **增量學習**：新增人物後立即生效，無需重新訓練
- **SQLite WAL**：多執行緒並發讀寫，不鎖死
