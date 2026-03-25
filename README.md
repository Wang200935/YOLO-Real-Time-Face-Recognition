<h1 align="center">🎯 YOLO Real-Time Face Recognition System</h1>

<p align="center">
  <strong>Production-grade real-time face recognition powered by YOLOv11 + MobileFaceNet</strong><br/>
  Runs on Windows · macOS · Linux · Raspberry Pi
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> ·
  <a href="docs/README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="docs/README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="docs/README_ja.md">🇯🇵 日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv11-Pose-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Win%20%7C%20Mac%20%7C%20Linux-informational?style=flat-square"/>
  <img src="https://img.shields.io/github/v/release/Wang200935/YOLO-Real-Time-Face-Recognition?style=flat-square"/>
</p>

---

## ⬇️ Download

Head to the **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** page to download the latest version:

| Platform | File |
|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` |
| 🐧 Linux / Developers | See [Run from Source](#run-from-source) below |

> Models are **automatically downloaded** (~22 MB total) on first launch.

---

## ✨ What It Does

| | |
|---|---|
| 🎯 **Instant detection** | Identifies known faces the moment they appear on camera |
| 🧠 **Self-improving** | Automatically learns new angles while the camera is running |
| 🔄 **360° coverage** | Collects front, side, and top-down angles so turning your head won't break recognition |
| 📸 **Smart capture** | Only saves sharp, high-quality frames — blurry ones are discarded |
| 🕴️ **Full-body scenes** | Works even when the full body is visible (not just close-up faces) |
| 🚫 **No ghost profiles** | Side profiles are matched to existing people rather than creating duplicate visitors |
| 🔔 **Alert system** | Set alerts for specific individuals |
| 🖥️ **Desktop GUI** | Clean interface with model switcher, threshold slider, and person management |

---

## 🚀 Run from Source

### Requirements

- Python **3.10+**  
- A USB or built-in camera  
- 4 GB RAM recommended

### Setup

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎓 Registering a Person

### Option A — Live Camera
1. Launch the app and click **▶ Start Recognition**
2. Stand in front of the camera for ~3 seconds
3. The system auto-registers you as a visitor and keeps learning your angles

### Option B — Upload Photos (Recommended)
1. Click **➕ Add Person** → enter a name → select photos
2. For best accuracy, provide **8–15 photos** covering:

| Angle | Tip |
|---|---|
| Front | Look directly at camera |
| Left / Right 45° | Turn head slightly to each side |
| Left / Right 90° | Full side profile |
| Chin up / down | Slight tilt upward and downward |
| Full body | Stand 2–3 m from camera |

---

## ⚙️ Settings

All behaviour can be tuned in **Settings** within the app, or by editing `config.py`:

| Setting | Default | Effect |
|---|---|---|
| Recognition Threshold | 0.64 | Lower = stricter matching |
| Skip Frames | 2 | Higher = faster, less smooth |
| Detection Confidence | 0.45 | Lower = picks up more distant faces |
| Detection Resolution | 640 | Lower (320) for weaker hardware |
| GPU Acceleration | Off | Set device to `cuda:0` to enable |

---

## ❓ Troubleshooting

**Camera won't open**
→ Another app may be using it. Try changing the camera index to `1` in settings.
→ macOS: grant camera permission in *System Settings → Privacy → Camera*

**Recognition is inaccurate**
→ Register with more angles (8–15 photos recommended)
→ Lower the recognition threshold in settings

**Low FPS**
→ Reduce detection resolution to 320
→ Increase skip frames to 4

**Model download fails**
→ Check your internet connection, or manually place the model files in the `data/models/` folder

---

## 📄 License

MIT License
