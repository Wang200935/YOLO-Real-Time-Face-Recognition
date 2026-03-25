<h1 align="center">🎯 YOLO 即時人臉辨識系統</h1>

<p align="center">
  <strong>基於 YOLOv11-Pose + MobileFaceNet 的生產級即時人臉辨識桌面應用</strong><br/>
  增量學習 · 360° 角度覆蓋 · 全身場景支援 · 防誤判自動建檔
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv11-Pose-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Win%20%7C%20Mac%20%7C%20Linux-informational?style=flat-square"/>
  <img src="https://img.shields.io/github/v/release/Wang200935/YOLO-Real-Time-Face-Recognition?style=flat-square"/>
</p>

---

## ⬇️ 下載

前往 **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** 下載最新版本（無需安裝 Python）：

| 平台 | 檔案 | 說明 |
|---|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` | 直接雙擊執行 |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` | 拖曳至應用程式資料夾 |
| 🐧 Linux / 開發者 | 見下方[一鍵安裝](#-一鍵安裝) | 需要 Python 3.10+ |

> AI 模型於**首次啟動時自動下載**（約 22 MB），無需手動設定。

---

## ✨ 功能特色

### 👤 人臉辨識
- 人臉一出現即立刻辨識已知人物，低延遲、無需 GPU
- 支援單幀中同時辨識多人
- 辨識距離可達數公尺

### 🧠 增量學習 & 360° 覆蓋
- 攝影機運行中自動收集特徵向量，每人最多 50 個
- 自動收集正面、側面、俯仰角等多角度向量
- 新角度出現時替換最冗餘的舊向量，實現全方位覆蓋
- **轉頭或低頭不再導致辨識失敗**

### 🕴️ 全身場景支援
- 全身入鏡時仍可準確擷取人臉區域並辨識
- 支援遠距離（數公尺）辨識
- 關鍵點引導的臉部擷取，在不穩定姿勢下仍準確

### 📸 智慧截圖篩選
- 使用拉普拉斯方差評估每幀清晰度
- 只保留品質最高的幀用於訓練
- 模糊幀自動丟棄，確保特徵品質

### 🚫 防誤判建檔
- 建立新訪客前進行**最終比對**，確認與已知人物的相似度
- 側臉（餘弦相似度約 0.2–0.5）會被歸類至已知人物而非重複建檔
- 徹底解決「同一個人被建立多個訪客」的問題

### 🔔 警報系統
- 右鍵任意人物 → 設定警示
- 偵測到該人物時播放音效提醒
- 適用於安全管控、考勤或門禁場景

### 🖥️ 圖形介面
- 簡潔 Tkinter 桌面介面
- **模型切換器**：可在 YOLOv11n/s/m/l/x 之間切換
- **閾值滑桿**：即時調整辨識靈敏度
- **人物管理**：新增、重新命名、刪除、查看縮圖
- **批量導入**：一次導入整個資料夾（每個子資料夾為一人）

---

## 🚀 一鍵安裝

### macOS / Linux

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
chmod +x install.sh && ./install.sh
```

安裝完成後啟動：
```bash
source venv/bin/activate
python main.py
```

### Windows

```bat
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
install.bat
```

安裝完成後啟動：
```bat
venv\Scripts\activate
python main.py
```

---

## 🎓 新增人物

### 方法 A — 即時鏡頭（最簡單）
1. 啟動應用程式並點擊 **▶ 啟動辨識**
2. 站在攝影機前約 3 秒
3. 系統自動建立訪客資料並持續在背景學習你的角度
4. 使用**重新命名**為訪客取一個正式名稱

### 方法 B — 上傳照片（精準度最佳）
1. 點擊 **➕ 新增人物** → 輸入姓名 → 選擇照片

建議提供 **8–15 張**，涵蓋以下角度：

| 角度 | 建議 |
|---|---|
| 正面 | 直視鏡頭，自然表情 |
| 左側 45° | 頭部向左微轉 |
| 右側 45° | 頭部向右微轉 |
| 左側全側面 | 完全側面（90°） |
| 右側全側面 | 完全側面（90°） |
| 微低頭 | 頭部略微向下傾 |
| 微抬頭 | 頭部略微向上仰 |
| 全身 | 距鏡頭 2–3 公尺，臉部清晰可見 |

### 方法 C — 批量導入
準備一個資料夾，**每個子資料夾 = 一個人**：
```
photos/
├── 王小明/  (img1.jpg, img2.jpg …)
└── 李小華/  (img1.jpg, img2.jpg …)
```
點擊 **📁 批量導入** 並選擇根資料夾。

---

## ⚙️ 設定參數

| 參數 | 預設值 | 說明 |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | 餘弦相似度閾值，越低越嚴格 |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | 每人最多特徵向量數 |
| `SKIP_FRAMES` | 2 | 每 N 幀偵測一次，越高越省效能 |
| `DETECT_CONFIDENCE` | 0.45 | YOLO 信心度，越低越能偵測遠距人臉 |
| `DETECT_IMG_SIZE` | 640 | 偵測解析度，低效能設備建議 320 |
| `DETECT_DEVICE` | `"cpu"` | 設為 `"cuda:0"` 啟用 NVIDIA GPU |
| `CAMERA_INDEX` | `0` | 攝影機裝置索引 |

---

## 🏗️ 技術架構

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   CameraThread  │────▶│  DetectionThread │────▶│ RecognitionThread │
│  (OpenCV 擷取)   │     │ (YOLOv11-Pose    │     │ (MobileFaceNet    │
│                 │     │  推理)            │     │  餘弦比對)         │
└─────────────────┘     └──────────────────┘     └────────┬──────────┘
                                                           │
                              ┌────────────────────────────┘
                              ▼
                    ┌──────────────────┐     ┌───────────────────────┐
                    │  GUI 主執行緒     │     │    DBWriterThread      │
                    │  (Tkinter 繪製 +  │     │  (SQLite WAL 非同步    │
                    │   增量學習觸發)    │     │   批次寫入)             │
                    └──────────────────┘     └───────────────────────┘
```

### 模組說明

| 模組 | 功能 |
|---|---|
| `main.py` | 程式入口，初始化所有元件 |
| `config.py` | 集中式設定，支援執行期動態修改 |
| `face_detector.py` | YOLOv11-Pose 推理、關鍵點擷取、臉部區域衍生 |
| `face_recognizer.py` | MobileFaceNet 嵌入向量、餘弦索引、增量新增/替換 |
| `database.py` | SQLite WAL 架構、Thread-local 連線、嵌入向量 CRUD |
| `gui.py` | Tkinter UI、攝影機迴圈、緩衝管理、增量學習觸發 |
| `worker_threads.py` | Camera/Detection/Recognition/DBWriter 執行緒管理 |
| `utils.py` | JPEG 編碼、圖像品質評分、雜項工具 |

---

## ❓ 常見問題

<details>
<summary><strong>攝影機無法開啟</strong></summary>

- 其他程式可能正在使用攝影機
- 在設定中將攝影機索引改為 `1` 或 `2`
- **macOS**：前往 系統設定 → 隱私與安全性 → 相機，授予 Python 存取權限
- **Linux**：執行 `sudo usermod -aG video $USER` 後重新登入
</details>

<details>
<summary><strong>辨識準確率不佳</strong></summary>

- 用更多不同角度的照片重新建檔（建議 8–15 張）
- 確保訓練照片光線充足且臉部清晰
- 在設定中將辨識閾值調低至 `0.55`
- 清除所有資料後重新建檔
</details>

<details>
<summary><strong>模型下載失敗</strong></summary>

- 確認網路連線正常
- 手動下載 `yolo11n-pose.pt` 並放入 `data/models/`
- 從 InsightFace 下載 `buffalo_sc.zip`，解壓取出 `w600k_mbf.onnx` 放入 `data/models/`
</details>

<details>
<summary><strong>FPS 過低 / CPU 使用率過高</strong></summary>

- 將跳幀數增加至 `4`
- 將偵測解析度降低至 `320`
- 降低攝影機解析度：`CAMERA_WIDTH=320, CAMERA_HEIGHT=240`
- 切換至更小的模型（yolo11n-pose 最快）
</details>

<details>
<summary><strong>Raspberry Pi 特定設定</strong></summary>

```bash
sudo apt install python3-tk libsdl2-mixer-2.0-0
pip install -r requirements.txt
```
Pi 4 建議使用 `DETECT_IMG_SIZE=320` 和 `SKIP_FRAMES=4`。
</details>

---

## 📄 授權

MIT License — 詳見 [LICENSE](../LICENSE)
