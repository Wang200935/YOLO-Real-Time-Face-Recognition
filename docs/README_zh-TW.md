<h1 align="center">🎯 YOLO 即時人臉辨識系統</h1>

<p align="center">
  <strong>基於 YOLOv11 + MobileFaceNet 的生產級即時人臉辨識桌面應用</strong><br/>
  支援 Windows · macOS · Linux · Raspberry Pi
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ⬇️ 下載

前往 **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** 下載最新版本：

| 平台 | 檔案 |
|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` |
| 🐧 Linux / 開發者 | 請見下方[從原始碼執行](#從原始碼執行) |

> 模型檔案將在**首次啟動時自動下載**（約 22 MB）。

---

## ✨ 功能特色

| | |
|---|---|
| 🎯 **即時辨識** | 人臉一出現即立刻辨識已知人物 |
| 🧠 **自我學習** | 攝影機運行中自動學習新角度 |
| 🔄 **360° 覆蓋** | 自動收集正面、側面、俯仰角，轉頭不再辨識失敗 |
| 📸 **智慧截圖** | 只儲存清晰高品質幀，模糊幀自動丟棄 |
| 🕴️ **全身場景** | 全身入鏡時仍可準確辨識人臉 |
| 🚫 **防重複建檔** | 側臉不會被誤判為新訪客而重複建立資料 |
| 🔔 **警報系統** | 可對特定人物設定警示提醒 |
| 🖥️ **桌面介面** | 簡潔 GUI，支援模型切換、閾值調整與人物管理 |

---

## 🚀 從原始碼執行

### 需求

- Python **3.10+**
- USB 或內建攝影機
- 建議 4 GB RAM

### 安裝

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎓 新增人物

### 方法 A — 即時鏡頭
1. 啟動應用程式並點擊 **▶ 啟動辨識**
2. 站在攝影機前約 3 秒
3. 系統自動建立訪客資料並持續學習你的各種角度

### 方法 B — 上傳照片（推薦）
1. 點擊 **➕ 新增人物** → 輸入姓名 → 選擇照片
2. 建議提供 **8–15 張**，涵蓋以下角度：

| 角度 | 建議 |
|---|---|
| 正面 | 直視鏡頭 |
| 左 / 右 45° | 頭部微轉 |
| 左 / 右 90° | 完全側面 |
| 微低頭 / 微抬頭 | 上下小幅傾斜 |
| 全身 | 距鏡頭 2–3 公尺站立 |

---

## ⚙️ 設定

所有行為可在應用程式的**設定**介面調整，或直接編輯 `config.py`：

| 設定項目 | 預設值 | 作用 |
|---|---|---|
| 辨識閾值 | 0.64 | 越低越嚴格 |
| 跳幀數 | 2 | 越高效能負擔越低 |
| 偵測信心度 | 0.45 | 越低越能偵測遠距人臉 |
| 偵測解析度 | 640 | 低效能裝置建議設 320 |
| GPU 加速 | 關閉 | 設為 `cuda:0` 啟用 |

---

## ❓ 常見問題

**攝影機無法開啟**
→ 其他程式可能正在使用攝影機，嘗試在設定中將攝影機索引改為 `1`
→ macOS：前往 系統設定 → 隱私與安全性 → 相機，授予存取權限

**辨識準確率不佳**
→ 用更多角度的照片重新建檔（建議 8–15 張）
→ 在設定中降低辨識閾值

**FPS 過低**
→ 將偵測解析度降至 320，跳幀數調高至 4

**模型下載失敗**
→ 確認網路連線，或手動將模型檔案放入 `data/models/` 資料夾

---

## 📄 授權

MIT License
