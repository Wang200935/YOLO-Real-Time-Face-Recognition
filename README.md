<h1 align="center">🎯 YOLO Real-Time Face Recognition System</h1>

<p align="center">
  <strong>A production-grade, real-time face recognition system powered by YOLOv11 + MobileFaceNet.</strong><br/>
  Incremental learning · Multi-angle coverage · Full-body detection support
</p>

<p align="center">
  <a href="README.md">🇬🇧 English</a> ·
  <a href="docs/README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="docs/README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="docs/README_ja.md">🇯🇵 日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/YOLOv11-Pose-green?style=flat-square&logo=ultralytics"/>
  <img src="https://img.shields.io/badge/MobileFaceNet-ONNX-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/SQLite-WAL-lightgrey?style=flat-square&logo=sqlite"/>
  <img src="https://img.shields.io/badge/Platform-Win%20%7C%20Mac%20%7C%20Linux-informational?style=flat-square"/>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Real-Time Detection** | YOLOv11-Pose detects faces and body keypoints simultaneously |
| 🧠 **Incremental Learning** | Auto-collects diverse angle vectors (front, side, top-down) while running |
| 🔄 **Angle-Diversity Replacement** | Replaces redundant vectors with new angles to ensure full 360° coverage |
| 📸 **Quality-Based Capture** | Only registers the sharpest frames; blurry frames are auto-discarded |
| 🕴️ **Full-Body Scene Support** | Detects and recognizes faces even when the full body is visible |
| 🚫 **Anti-False-Registration** | Final cross-check before registering a new unknown visitor |
| ⚡ **Multi-Thread Pipeline** | Camera → Detect → Recognize → DB Writer, each on its own thread |
| 🗄️ **SQLite WAL** | Concurrent multi-thread read/write without locks |
| 🖥️ **GUI** | Tkinter-based interface with model switcher, threshold slider, and alert system |

---

## 🚀 Quick Start

### Requirements

- Python **3.10+**
- USB or built-in camera
- RAM: 4 GB+
- OS: Windows 10+, macOS 12+, Ubuntu 20.04+, Raspberry Pi OS (64-bit)

### Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

On first run, the system **automatically downloads**:
- `yolo11n-pose.pt` (~6 MB) — YOLO detection model
- `w600k_mbf.onnx` (~16 MB) — MobileFaceNet recognition model

---

## 🎓 How to Register a Person

### Via Camera (Best for full-body scenes)
1. Click **▶ Start Recognition** and stand in front of the camera.
2. After 3 seconds, the system will automatically register you as a visitor.
3. Go to **➕ Add Person** to rename the visitor profile.

### Via Photos (Recommended for precise recognition)
1. Click **➕ Add Person** → enter name → select photos.
2. For best results, provide **8–15 photos** covering these angles:

| Angle | Description |
|---|---|
| Front | Looking directly at camera |
| Left 45° | Head turned ~45° left |
| Right 45° | Head turned ~45° right |
| Left 90° | Full left profile |
| Right 90° | Full right profile |
| Chin down | Head slightly tilted down |
| Chin up | Head slightly tilted up |
| Full body | 2–3 meters away, face clearly visible |

---

## ⚙️ Configuration

Edit `config.py` to tune performance:

| Parameter | Default | Description |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | Similarity threshold (lower = stricter) |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | Max feature vectors per person |
| `SKIP_FRAMES` | 2 | Detection interval (higher = faster but less smooth) |
| `DETECT_CONFIDENCE` | 0.45 | YOLO detection confidence |
| `DETECT_IMG_SIZE` | 640 | Detection resolution (320 for Pi) |
| `DETECT_DEVICE` | "cpu" | Change to "cuda:0" for GPU |
| `CAMERA_INDEX` | 0 | Camera device index |

---

## 🏗️ Architecture

```
CameraThread ──► DetectionThread ──► RecognitionThread ──► GUI Main Thread
 (OpenCV)         (YOLOv11-Pose)     (MobileFaceNet)        (Tkinter)
                                            ↓
                                     DBWriterThread (SQLite WAL)
```

- **YOLOv11-Pose**: End-to-end inference without NMS, optimized for CPU
- **MobileFaceNet**: 512-dim cosine similarity recognition
- **Incremental Learning**: Triggered every 1s for active tracked faces
- **Angle-Diversity Guard**: Similarity matrix prevents duplicate angle vectors

---

## 📁 Project Structure

```
YOLO-Real-Time-Face-Recognition/
├── main.py              # Entry point
├── config.py            # Configuration
├── face_detector.py     # YOLOv11 pose detection
├── face_recognizer.py   # MobileFaceNet + FAISS-like index
├── database.py          # SQLite WAL database manager
├── gui.py               # Tkinter GUI + incremental learning logic
├── worker_threads.py    # Multi-thread pipeline
├── utils.py             # Utility functions
├── requirements.txt
├── docs/
│   ├── README_zh-TW.md  # 繁體中文
│   ├── README_zh-CN.md  # 简体中文
│   └── README_ja.md     # 日本語
└── data/                # Auto-created on first run
    ├── models/          # AI model files
    ├── db/              # SQLite database
    ├── screenshots/     # Person thumbnails
    └── logs/            # Runtime logs
```

---

## ❓ FAQ

**Camera won't open?**
- Ensure no other app is using the camera
- Try `CAMERA_INDEX = 1` or `2` in `config.py`
- macOS: Grant camera permission in System Settings → Privacy

**Model download fails?**
- Check your internet connection
- Manually place `yolo11n-pose.pt` into `data/models/`
- Manually extract `w600k_mbf.onnx` from `buffalo_sc.zip` into `data/models/`

**Recognition accuracy is low?**
- Add more training photos with varied angles (8–15 recommended)
- Clear all data and re-register with more diverse angles
- Lower `RECOGNITION_THRESHOLD` to `0.55`

**Low FPS?**
- Increase `SKIP_FRAMES` to `4`
- Reduce `DETECT_IMG_SIZE` to `320`
- Reduce camera resolution: `CAMERA_WIDTH=320, CAMERA_HEIGHT=240`

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
