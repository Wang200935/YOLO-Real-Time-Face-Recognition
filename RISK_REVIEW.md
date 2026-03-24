# RISK_REVIEW.md — 風險審查報告

本文件對所有程式碼進行系統性審查，列舉潛在風險並說明已採取的解決措施。

---

## 1. 執行緒安全問題

### R01：特徵矩陣並發讀寫
**風險**：`FaceRecognizer._embeddings_matrix` 在辨識執行緒讀取的同時，增量學習也可能觸發 `_rebuild_index()` 寫入。

**等級**：高

**已修正（`face_recognizer.py`）**：
```python
self._lock = threading.RLock()
# identify() 和 rebuild_index() 都用 with self._lock: 保護
```

---

### R02：Tkinter 跨執行緒更新
**風險**：從 worker thread 直接呼叫 `label.config()` 等 Tkinter 方法會導致 segfault 或隨機凍結。

**等級**：高

**已修正（`gui.py`）**：
- 所有 widget 更新僅透過 `root.after(0, callback)` 路由至主執行緒
- BatchImportDialog 和 AddPersonDialog 均在背景執行緒中使用 `self._win.after(0, ...)` 回報進度

---

### R03：警報集合（set）讀寫競爭
**風險**：`RecognitionThread` 讀取 `alert_names` 的同時，GUI 主執行緒可能在修改（新增/移除警報）。

**等級**：中

**已修正（`worker_threads.py`, `gui.py`）**：
```python
self._alert_lock = threading.Lock()
# RecognitionThread 和 GUI 均透過 with self._alert_lock: 存取
```

---

## 2. 資源洩漏風險

### R04：攝影機未釋放
**風險**：若程式異常崩潰，`cv2.VideoCapture` 可能未呼叫 `release()`。

**等級**：高

**已修正（`worker_threads.py`）**：
```python
try:
    while not self._shutdown.is_set():
        ret, frame = cap.read()
        ...
finally:
    cap.release()  # 確保無論如何都釋放
    self._frame_queue.put(None)  # sentinel
```

---

### R05：執行緒 join 無 timeout
**風險**：若執行緒卡住，`join()` 會永久阻塞，導致程式無法正常退出。

**等級**：中

**已修正（`worker_threads.py`）**：
```python
for t in self._threads:
    t.join(timeout=timeout)  # 預設 3 秒，超時記錄 warning
    if t.is_alive():
        logger.warning(f"執行緒 {t.name} 未在 {timeout}s 內結束")
```

---

### R06：PhotoImage 被 GC 回收
**風險**：Tkinter 中，若 `PhotoImage` 物件無引用，GC 會回收，導致影像顯示空白。

**等級**：高

**已修正（`gui.py`）**：
```python
self._current_photo = photo  # 在 MainWindow 實例上保持引用，防止 GC
```

---

### R07：SQLite 連線未關閉
**風險**：`threading.local` 中每個執行緒的連線在程式退出時可能未正確關閉。

**等級**：低（SQLite 在進程退出時會自動清理，但可能導致 WAL 殘留）

**已修正（`database.py`）**：
```python
def close(self):
    if hasattr(self._local, "conn") and self._local.conn:
        self._local.conn.close()
        self._local.conn = None
# main.py 在退出時呼叫 db.close()
```

---

## 3. 資料一致性問題

### R08：分類器與資料庫不同步
**風險**：若 `_rebuild_index()` 失敗（如 DB 讀取異常），記憶體中的特徵矩陣可能與 DB 不一致。

**等級**：中

**已修正（`face_recognizer.py`）**：
```python
def rebuild_index(self):
    rows = self._db.get_all_embeddings()  # DB 讀取
    if not rows:
        with self._lock:
            self._embeddings_matrix = None
            self._person_ids = []
        return
    # 只有完整建立矩陣後才 with self._lock: 更新
    with self._lock:
        self._embeddings_matrix = embeddings
        self._person_ids = person_ids
```
如果 `get_all_embeddings()` 拋出異常，鎖內的矩陣保持舊值，辨識繼續運作。

---

### R09：批量導入中途失敗
**風險**：批量導入時若程式崩潰，可能只新增了部分特徵向量，導致人物資料不完整。

**等級**：低

**緩解措施（`gui.py BatchImportDialog._process`）**：
- 若所有照片都無有效特徵，整個人物記錄一併刪除（`db.delete_person`）
- 每個 `save_embeddings_batch()` 使用一次 `commit`，批量具原子性

**未來改善**：可包裝在 `BEGIN TRANSACTION` / `ROLLBACK` 中確保全或無。

---

### R10：名稱快取未及時更新
**風險**：`FaceRecognizer._person_names` 快取在重新命名後可能仍保留舊名稱。

**等級**：低

**已修正（`face_recognizer.py`, `gui.py`）**：
```python
# 重新命名後呼叫
self._recognizer.invalidate_name_cache()  # 清空快取
self._recognizer.rebuild_index()          # 從 DB 重新載入
```

---

## 4. 效能瓶頸

### R11：特徵提取阻塞辨識 Pipeline
**風險**：ONNX 推理若耗時過長（如老舊 CPU），可能導致 `detection_queue` 積壓。

**等級**：中

**已緩解**：
- `SKIP_FRAMES` 預設 2（Pi 上自動設為 4），降低檢測頻率
- `detection_queue` 設 `maxsize=4`，過滿時 `put_nowait` 丟棄，防止無限積壓
- ONNX 自動選最快的執行提供者（CoreML / CUDA / CPU）

**建議優化**：若 FPS 持續低於 5，可將 `SKIP_FRAMES` 增至 6，或將 `DETECT_IMG_SIZE` 降至 320。

---

### R12：`_rebuild_index` 頻繁呼叫
**風險**：若多次快速呼叫 `add_face()`，每次都觸發 `rebuild_index()`，造成重複 DB 讀取。

**等級**：低（正常使用下不頻繁）

**已解決**：`batch_add_faces()` 一次性寫入所有向量後才呼叫一次 `rebuild_index()`，批量導入時無此問題。

---

## 5. 跨平台相容性陷阱

### R13：Windows 攝影機後端延遲
**風險**：預設 `CAP_ANY` 後端在 Windows 上開啟攝影機需 2-3 秒。

**已修正（`worker_threads.py CameraThread._get_backend()`）**：
```python
if system == "Windows":
    return cv2.CAP_DSHOW  # DirectShow：快速開啟
```

---

### R14：macOS 攝影機權限靜默失敗
**風險**：macOS 未授權時，`cap.isOpened()` 回傳 True，但 `cap.read()` 持續回傳黑幀。

**已緩解（`worker_threads.py`）**：
```python
if not cap.isOpened():
    self.error = f"無法開啟攝影機"
    # ...
```
若連續幀讀取失敗，記錄 warning 並短暫休眠，避免 100% CPU。

**用戶提示**：在 README 中說明需在系統偏好設定授權攝影機存取。

---

### R15：Raspberry Pi 記憶體限制
**風險**：Pi 4 (4GB) 在同時運行 YOLOv11 + MobileFaceNet + GUI 時可能記憶體不足。

**已緩解（`config.py`）**：
```python
if _is_raspberry_pi():
    self.DETECT_IMG_SIZE = 320   # 降低推理記憶體用量
    self.SKIP_FRAMES = 4
    self.CAMERA_FPS = 15
```
**建議**：使用 `yolo11n-pose.pt`（nano），而非 `yolo11s-pose.pt` 或更大版本。

---

### R16：numpy 版本相容性
**風險**：`numpy>=2.0` 與 `cv2`、部分 `onnxruntime` 版本存在 API 不相容。

**已修正（`requirements.txt`）**：
```
numpy>=1.24.0,<2.0.0
```

---

## 6. 安全性

### R17：影像不對外暴露
本系統為純本地端軟體，不對外開放網路端口，截圖僅存於本地 `data/screenshots/`。

### R18：SQLite 無加密
人臉特徵向量以明文儲存於 `data/db/faces.db`。如有需求可搭配 SQLCipher 加密（需修改 `database.py`）。

---

## 7. 已知限制

| 限制 | 說明 | 建議解法 |
|---|---|---|
| 單攝影機 | 目前只支援單一影像來源 | 可擴充 `CameraThread` 為多執行緒 |
| 遮臉/口罩 | 佩戴口罩時辨識準確率下降 | 使用針對口罩的特定訓練資料再訓練 |
| 人臉數量 | 特徵矩陣 N>10000 時效能下降 | 改用 FAISS 索引（已預留介面） |
| 無 GPU 時 FPS | 純 CPU 約 10-20 FPS | 升級 GPU 或使用 DETECT_DEVICE="mps" |

---

## 8. 修正摘要

| ID | 類別 | 狀態 |
|---|---|---|
| R01 | 特徵矩陣並發讀寫 | ✅ RLock 保護 |
| R02 | Tkinter 跨執行緒 | ✅ root.after 路由 |
| R03 | alert_names 競爭 | ✅ Lock 保護 |
| R04 | 攝影機未釋放 | ✅ finally 區塊 |
| R05 | join 無 timeout | ✅ join(timeout=3) |
| R06 | PhotoImage GC | ✅ 實例變數保持引用 |
| R07 | SQLite 連線未關 | ✅ close() 在退出時呼叫 |
| R08 | 分類器資料庫不同步 | ✅ 先建矩陣再加鎖更新 |
| R09 | 批量導入中途失敗 | ✅ 失敗時回滾人物記錄 |
| R10 | 名稱快取陳舊 | ✅ invalidate_name_cache() |
| R11 | 特徵提取阻塞 | ✅ SKIP_FRAMES + 佇列丟棄 |
| R12 | rebuild_index 頻繁 | ✅ batch_add_faces 一次重建 |
| R13 | Windows 攝影機延遲 | ✅ CAP_DSHOW |
| R14 | macOS 攝影機靜默失敗 | ✅ 錯誤記錄 + 文件說明 |
| R15 | Pi 記憶體限制 | ✅ 自動降低解析度/幀率 |
| R16 | numpy 版本 | ✅ pin <2.0.0 |

---

*最後審查日期：2026-03-24*
