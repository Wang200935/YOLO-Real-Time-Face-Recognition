<h1 align="center">🎯 YOLO 实时人脸识别系统</h1>

<p align="center">
  <strong>基于 YOLOv11-Pose + MobileFaceNet 的生产级实时人脸识别桌面应用</strong><br/>
  增量学习 · 360° 角度覆盖 · 全身场景支持 · 防误判自动建档
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ⬇️ 下载

前往 **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** 下载最新版本（无需安装 Python）：

| 平台 | 文件 | 说明 |
|---|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` | 直接双击运行 |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` | 拖拽至应用程序文件夹 |
| 🐧 Linux / 开发者 | 见下方[一键安装](#-一键安装) | 需要 Python 3.10+ |

> AI 模型于**首次启动时自动下载**（约 22 MB），无需手动配置。

---

## ✨ 功能特色

### 👤 人脸识别
- 人脸出现即刻识别已知人物，低延迟、无需 GPU
- 支持单帧中同时识别多人
- 识别距离可达数米

### 🧠 增量学习 & 360° 覆盖
- 摄像头运行中自动收集特征向量，每人最多 50 个
- 自动收集正面、侧面、俯仰角等多角度向量
- 新角度出现时替换最冗余的旧向量，实现全方位覆盖
- **转头或低头不再导致识别失败**

### 🕴️ 全身场景支持
- 全身入镜时仍可准确提取人脸区域并识别
- 支持远距离（数米）识别
- 关键点引导的人脸提取，在不稳定姿势下仍准确

### 📸 智能截图筛选
- 使用拉普拉斯方差评估每帧清晰度
- 只保留质量最高的帧用于训练
- 模糊帧自动丢弃，确保特征质量

### 🚫 防误判建档
- 建立新访客前进行**最终比对**，确认与已知人物的相似度
- 侧脸（余弦相似度约 0.2–0.5）会被归类至已知人物而非重复建档
- 彻底解决「同一个人被创建多个访客」的问题

### 🔔 警报系统
- 右键任意人物 → 设置警示
- 检测到该人物时播放音效提醒
- 适用于安全管控、考勤或门禁场景

### 🖥️ 图形界面
- 简洁 Tkinter 桌面界面
- **模型切换器**：可在 YOLOv11n/s/m/l/x 之间切换
- **阈值滑块**：实时调整识别灵敏度
- **人员管理**：添加、重命名、删除、查看缩略图
- **批量导入**：一次导入整个文件夹（每个子文件夹为一人）

---

## 🚀 一键安装

### macOS / Linux

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
chmod +x install.sh && ./install.sh
```

安装完成后启动：
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

安装完成后启动：
```bat
venv\Scripts\activate
python main.py
```

---

## 🎓 添加人员

### 方法 A — 实时摄像头（最简单）
1. 启动应用并点击 **▶ 启动识别**
2. 在摄像头前站立约 3 秒
3. 系统自动建立访客档案并持续在后台学习各角度
4. 使用**重命名**为访客取一个正式名称

### 方法 B — 上传照片（精度最佳）
建议提供 **8–15 张**，涵盖以下角度：

| 角度 | 建议 |
|---|---|
| 正面 | 直视镜头，自然表情 |
| 左侧 45° | 头部向左微转 |
| 右侧 45° | 头部向右微转 |
| 左侧全侧面 | 完全侧面（90°） |
| 右侧全侧面 | 完全侧面（90°） |
| 微低头 | 头部略微向下倾 |
| 微抬头 | 头部略微向上仰 |
| 全身 | 距摄像头 2–3 米，脸部清晰可见 |

### 方法 C — 批量导入
准备一个文件夹，**每个子文件夹 = 一个人**：
```
photos/
├── 张三/  (img1.jpg, img2.jpg …)
└── 李四/  (img1.jpg, img2.jpg …)
```
点击 **📁 批量导入** 并选择根文件夹。

---

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | 余弦相似度阈值，越低越严格 |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | 每人最多特征向量数 |
| `SKIP_FRAMES` | 2 | 每 N 帧检测一次，越高越省性能 |
| `DETECT_CONFIDENCE` | 0.45 | YOLO 置信度，越低越能检测远距人脸 |
| `DETECT_IMG_SIZE` | 640 | 检测分辨率，低性能设备建议 320 |
| `DETECT_DEVICE` | `"cpu"` | 设为 `"cuda:0"` 启用 NVIDIA GPU |
| `CAMERA_INDEX` | `0` | 摄像头设备索引 |

---

## 🏗️ 技术架构（开发者）

```
CameraThread ──► DetectionThread ──► RecognitionThread
 (OpenCV 采集)   (YOLOv11-Pose 推理)  (MobileFaceNet 比对)
                                           │
                         ┌─────────────────┘
                         ▼
               GUI 主线程 (Tkinter + 增量学习触发)
               DBWriterThread (SQLite WAL 异步写入)
```

| 模块 | 功能 |
|---|---|
| `face_detector.py` | YOLOv11-Pose 推理、关键点提取、人脸区域衍生 |
| `face_recognizer.py` | MobileFaceNet 嵌入向量、余弦索引、增量添加/替换 |
| `database.py` | SQLite WAL 架构、线程本地连接、嵌入向量 CRUD |
| `gui.py` | Tkinter UI、缓冲管理、增量学习触发 |
| `worker_threads.py` | 各执行线程管理 |

---

## ❓ 常见问题

<details>
<summary><strong>摄像头无法打开</strong></summary>

- 确认其他程序未占用摄像头
- 将摄像头索引改为 `1` 或 `2`
- **macOS**：系统设置 → 隐私与安全性 → 摄像头，授予权限
- **Linux**：执行 `sudo usermod -aG video $USER`
</details>

<details>
<summary><strong>识别准确率低</strong></summary>

- 用更多不同角度的照片重新注册（建议 8–15 张）
- 将识别阈值调低至 `0.55`
- 清除数据后重新注册
</details>

<details>
<summary><strong>模型下载失败</strong></summary>

- 确认网络连接正常
- 手动下载模型文件放入 `data/models/`
</details>

<details>
<summary><strong>FPS 低 / CPU 占用高</strong></summary>

- 跳帧数增加至 `4`，检测分辨率降至 `320`
- 降低摄像头分辨率：`CAMERA_WIDTH=320, CAMERA_HEIGHT=240`
</details>

