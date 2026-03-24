<h1 align="center">🎯 YOLO 实时人脸识别系统</h1>

<p align="center">
  <strong>基于 YOLOv11 + MobileFaceNet 的生产级实时人脸识别系统</strong><br/>
  增量学习 · 全方位角度覆盖 · 全身场景支持
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ✨ 功能特色

| 功能 | 说明 |
|---|---|
| 🎯 **实时检测** | YOLOv11-Pose 同时检测人脸与姿态关键点 |
| 🧠 **增量学习** | 运行中自动采集多角度特征向量（正面、侧面、俯仰） |
| 🔄 **角度多样性替换** | 自动淘汰冗余向量，确保 360° 全方位覆盖 |
| 📸 **质量筛选截图** | 仅保留最清晰帧，自动排除模糊帧 |
| 🕴️ **全身场景支持** | 全身出现时仍可准确检测并识别人脸 |
| 🚫 **防误判自动注册** | 侧脸等角度不会被误认为陌生人并重复建档 |
| ⚡ **多线程 Pipeline** | 摄像头→检测→识别→数据库，各自独立线程 |
| 🗄️ **SQLite WAL** | 多线程并发读写，不锁死 |
| 🖥️ **图形界面** | 支持模型切换、阈值调整、警报设定 |

---

## 🚀 快速开始

### 系统要求

- Python **3.10+**
- USB 或内置摄像头
- RAM：4 GB+
- 操作系统：Windows 10+, macOS 12+, Ubuntu 20.04+, Raspberry Pi OS (64-bit)

### 安装步骤

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动程序
python main.py
```

首次运行时，程序将自动下载：
- `yolo11n-pose.pt`（约 6 MB）— YOLO 检测模型
- `w600k_mbf.onnx`（约 16 MB）— MobileFaceNet 识别模型

---

## 🎓 如何添加人员

### 通过摄像头（适合全身场景）
1. 点击 **▶ 启动识别**，站在摄像头前方。
2. 约 3 秒后系统自动建立访客档案。
3. 在 **➕ 添加人员** 中为访客重命名。

### 通过照片（推荐用于精准识别）
1. 点击 **➕ 添加人员** → 输入姓名 → 选择照片。
2. 建议提供 **8–15 张**，涵盖以下角度：

| 角度 | 说明 |
|---|---|
| 正面 | 直视镜头，自然表情 |
| 左侧 45° | 向左转约 45 度 |
| 右侧 45° | 向右转约 45 度 |
| 左侧 90° | 完全侧面 |
| 右侧 90° | 完全侧面 |
| 微低头 | 头略微下倾 |
| 微抬头 | 头略微上仰 |
| 全身 | 距摄像头 2–3 米，面部清晰可见 |

---

## ⚙️ 参数配置

修改 `config.py` 调整性能：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | 相似度阈值（越低越严格） |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | 每人最多特征向量数 |
| `SKIP_FRAMES` | 2 | 每 N 帧检测一次（越大越省性能）|
| `DETECT_CONFIDENCE` | 0.45 | YOLO 检测置信度阈值 |
| `DETECT_IMG_SIZE` | 640 | 检测分辨率（Pi 建议 320） |
| `DETECT_DEVICE` | "cpu" | "cuda:0" 启用 GPU |
| `CAMERA_INDEX` | 0 | 摄像头设备编号 |

---

## ❓ 常见问题

**摄像头无法打开？**
- 确认无其他程序占用摄像头
- 尝试将 `config.py` 中 `CAMERA_INDEX` 改为 `1` 或 `2`
- macOS：在系统设置 → 隐私与安全性 → 摄像头中授权

**模型下载失败？**
- 确认网络连接正常
- 手动将 `yolo11n-pose.pt` 放入 `data/models/`
- 从 `buffalo_sc.zip` 解压 `w600k_mbf.onnx` 至 `data/models/`

**识别准确率低？**
- 提供更多不同角度照片（建议 8–15 张）
- 清除现有数据后用多角度照片重新注册
- 将 `RECOGNITION_THRESHOLD` 降低至 `0.55`

---

## 📄 授权

MIT License — 详见 [LICENSE](../LICENSE)
