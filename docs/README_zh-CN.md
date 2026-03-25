<h1 align="center">🎯 YOLO 实时人脸识别系统</h1>

<p align="center">
  <strong>基于 YOLOv11 + MobileFaceNet 的生产级实时人脸识别桌面应用</strong><br/>
  支持 Windows · macOS · Linux · Raspberry Pi
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ⬇️ 下载

前往 **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** 下载最新版本：

| 平台 | 文件 |
|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` |
| 🐧 Linux / 开发者 | 请见下方[从源码运行](#从源码运行) |

> 模型文件将在**首次启动时自动下载**（约 22 MB）。

---

## ✨ 功能特色

| | |
|---|---|
| 🎯 **即时识别** | 人脸出现即刻识别已知人物 |
| 🧠 **自我学习** | 摄像头运行中自动学习新角度 |
| 🔄 **360° 覆盖** | 自动收集正面、侧面、俯仰角，转头不再识别失败 |
| 📸 **智能截图** | 只保存清晰高质量帧，模糊帧自动丢弃 |
| 🕴️ **全身场景** | 全身入镜时仍可准确识别人脸 |
| 🚫 **防重复建档** | 侧脸不会被误判为新访客而重复创建数据 |
| 🔔 **警报系统** | 可对特定人物设置警示提醒 |
| 🖥️ **桌面界面** | 简洁 GUI，支持模型切换、阈值调整与人员管理 |

---

## 🚀 从源码运行

### 要求

- Python **3.10+**
- USB 或内置摄像头
- 推荐 4 GB RAM

### 安装

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎓 添加人员

### 方法 A — 实时摄像头
1. 启动应用并点击 **▶ 启动识别**
2. 在摄像头前站立约 3 秒
3. 系统自动建立访客档案并持续学习各种角度

### 方法 B — 上传照片（推荐）
1. 点击 **➕ 添加人员** → 输入姓名 → 选择照片
2. 建议提供 **8–15 张**，涵盖以下角度：

| 角度 | 建议 |
|---|---|
| 正面 | 直视镜头 |
| 左 / 右 45° | 头部微转 |
| 左 / 右 90° | 完全侧面 |
| 微低头 / 微抬头 | 上下小幅倾斜 |
| 全身 | 距摄像头 2–3 米站立 |

---

## ⚙️ 设置

所有行为可在应用的**设置**界面调整：

| 设置项目 | 默认值 | 作用 |
|---|---|---|
| 识别阈值 | 0.64 | 越低越严格 |
| 跳帧数 | 2 | 越高性能负担越低 |
| 检测置信度 | 0.45 | 越低越能检测远距人脸 |
| 检测分辨率 | 640 | 低性能设备建议设 320 |
| GPU 加速 | 关闭 | 设为 `cuda:0` 启用 |

---

## ❓ 常见问题

**摄像头无法打开** → 其他程序可能占用，尝试将摄像头索引改为 `1`；macOS 请在系统设置中授予权限

**识别准确率低** → 用更多角度的照片重新注册（建议 8–15 张）；降低识别阈值

**FPS 低** → 将检测分辨率降至 320，跳帧数调高至 4

**模型下载失败** → 检查网络连接，或手动将模型文件放入 `data/models/` 文件夹

---

## 📄 许可

MIT License
