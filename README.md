<h1 align="center">🎯 YOLO Real-Time Face Recognition System</h1>

<p align="center">
  <strong>Production-grade real-time face recognition powered by YOLOv11-Pose + MobileFaceNet</strong><br/>
  Incremental learning · 360° angle coverage · Full-body scene support · Anti-false-registration
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
  <img src="https://img.shields.io/badge/MobileFaceNet-ONNX-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/SQLite-WAL-lightgrey?style=flat-square&logo=sqlite"/>
  <img src="https://img.shields.io/badge/Platform-Win%20%7C%20Mac%20%7C%20Linux-informational?style=flat-square"/>
  <img src="https://img.shields.io/github/v/release/Wang200935/YOLO-Real-Time-Face-Recognition?style=flat-square"/>
  <img src="https://img.shields.io/github/license/Wang200935/YOLO-Real-Time-Face-Recognition?style=flat-square"/>
</p>

---

## ⬇️ Download

Head to the **[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** page to download the latest pre-built installer — no Python required:

| Platform | File | Notes |
|---|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` | Double-click to run |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` | Drag to Applications |
| 🐧 Linux / Developers | See [One-Click Install](#-one-click-install) below | Python 3.10+ required |

> AI models are **automatically downloaded (~22 MB)** on first launch. No manual setup needed.

---

## ✨ Features

### 👤 Face Recognition
- Identifies registered people the instant they appear on camera
- Works in real-time at low latency on CPU (no GPU required)
- Supports multiple people in the same frame simultaneously

### 🧠 Incremental Learning & 360° Coverage
- Automatically collects feature vectors while the camera is running
- Builds a **50-vector profile** per person covering frontal, side, and top-down angles
- When new angles are detected, redundant vectors are replaced to maximize diversity
- Turning your head or tilting your chin **no longer breaks recognition**

### 🕴️ Full-Body Scene Support
- Detects and recognizes faces even when the full body is visible in frame
- Works at distances up to several meters from the camera
- Keypoint-guided face extraction ensures accuracy even in unstable poses

### 📸 Smart Frame Capture
- Rates each frame by sharpness (Laplacian variance)
- Only the highest-quality frames are stored for training
- Blurry or low-resolution frames are automatically discarded

### 🚫 Anti-False-Registration
- Before creating a new visitor profile, performs a **final cross-check** against all known people
- Side profiles (cosine similarity ~0.2–0.5 vs frontal) are matched instead of duplicated
- Eliminates the problem of one person appearing as multiple "Visitor" profiles

### 🔔 Alert System
- Right-click any person → set alert
- Plays a sound notification when that person is detected on camera
- Useful for security, attendance, or access control use cases

### 🖥️ GUI & Controls
- Clean Tkinter desktop interface
- **Model switcher**: Choose between YOLOv11n / s / m / l / x pose models
- **Threshold slider**: Adjust recognition sensitivity in real time
- **Person management**: Add, rename, delete, view thumbnails
- **Batch import**: Import a whole folder of images at once (one subfolder per person)

---

## 🚀 One-Click Install

### macOS / Linux

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
chmod +x install.sh && ./install.sh
```

Then launch:
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

Then launch:
```bat
venv\Scripts\activate
python main.py
```

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎓 Registering People

### Option A — Live Camera (Easiest)
1. Launch the app and click **▶ Start Recognition**
2. Stand in front of the camera for ~3 seconds
3. The system auto-registers you as a visitor and continuously learns your angles in the background
4. Use **Rename** to give the visitor a proper name

### Option B — Upload Photos (Recommended for Accuracy)
1. Click **➕ Add Person** → enter a name → select photos

For best recognition accuracy, provide **8–15 photos** covering these angles:

| Angle | Description |
|---|---|
| Front | Looking directly at the camera, neutral expression |
| Left 45° | Head turned ~45° to the left |
| Right 45° | Head turned ~45° to the right |
| Left Profile | Full left side view (90°) |
| Right Profile | Full right side view (90°) |
| Looking down | Head tilted slightly downward |
| Looking up | Head tilted slightly upward |
| Full body | Standing ~2–3 m from the camera |

### Option C — Batch Import
1. Prepare a folder where **each subfolder = one person's name**
   ```
   photos/
   ├── Alice/  (img1.jpg, img2.jpg …)
   └── Bob/    (img1.jpg, img2.jpg …)
   ```
2. Click **📁 Batch Import** and select the root folder

---

## ⚙️ Configuration

All settings are available in the **Settings** dialog inside the app, or you can edit `config.py` directly:

| Parameter | Default | Description |
|---|---|---|
| `RECOGNITION_THRESHOLD` | 0.64 | Cosine similarity threshold. Lower = stricter matching |
| `MAX_EMBEDDINGS_PER_PERSON` | 50 | Maximum feature vectors stored per person |
| `SKIP_FRAMES` | 2 | Process every Nth frame. Higher = less CPU load |
| `DETECT_CONFIDENCE` | 0.45 | YOLO detection confidence. Lower = detects more (farther) faces |
| `DETECT_IMG_SIZE` | 640 | Detection resolution. Use 320 for weaker hardware |
| `DETECT_DEVICE` | `"cpu"` | Set to `"cuda:0"` to use NVIDIA GPU |
| `CAMERA_INDEX` | `0` | Camera device index |
| `CAMERA_WIDTH` | `640` | Camera capture width |
| `CAMERA_HEIGHT` | `480` | Camera capture height |
| `ALERT_SOUND` | `True` | Enable sound alerts |

---

## 🏗️ Architecture (For Developers)

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   CameraThread  │────▶│  DetectionThread  │────▶│ RecognitionThread │
│  (OpenCV grab)  │     │ (YOLOv11-Pose    │     │ (MobileFaceNet   │
│                 │     │  inference)       │     │  cosine match)    │
└─────────────────┘     └──────────────────┘     └────────┬──────────┘
                                                           │
                              ┌────────────────────────────┘
                              ▼
                    ┌──────────────────┐     ┌───────────────────────┐
                    │  GUI Main Thread │     │    DBWriterThread     │
                    │  (Tkinter draw + │     │  (SQLite WAL async    │
                    │  incremental     │     │   batch writes)       │
                    │   learning)      │     └───────────────────────┘
                    └──────────────────┘
```

### Key Design Decisions

| Component | Choice | Reason |
|---|---|---|
| Detection | YOLOv11-Pose | Simultaneous body keypoint + face region extraction |
| Recognition | MobileFaceNet ONNX | 512-dim embeddings, fast on CPU, high accuracy |
| Similarity | Cosine distance | Scale-invariant, works well with normalized face embeddings |
| Diversity | Similarity matrix replacement | Prevents 50 near-identical frontal vectors, ensures angular coverage |
| Storage | SQLite WAL mode | Multi-thread safe, no ORM overhead, portable |
| Threading | Thread-per-stage pipeline | Decouples slow stages; each stage blocks independently |

### Module Overview

| Module | Role |
|---|---|
| `main.py` | Entry point, initializes all components |
| `config.py` | Centralized configuration with runtime-mutable parameters |
| `face_detector.py` | YOLOv11-Pose inference, keypoint extraction, face region derivation |
| `face_recognizer.py` | MobileFaceNet embedding, cosine index, incremental add/replace |
| `database.py` | SQLite WAL schema, thread-local connections, embedding CRUD |
| `gui.py` | Tkinter UI, camera loop, buffer management, incremental learning trigger |
| `worker_threads.py` | Camera/Detection/Recognition/DBWriter thread management |
| `utils.py` | JPEG encoding, image quality scoring, misc helpers |

### Incremental Learning Flow

```
Frame arrives → face detected → known person?
    │ YES → quality check (≥60) → interval check (≥1s)
    │         → add_face() → diversity check
    │           → if capacity full AND new angle (sim < 0.80)
    │               → replace most redundant vector
    │
    │ NO  → buffer_unknown()
    │         → collect N frames → quality filter
    │           → final cross-check (sim ≥ 0.20) against all known
    │               → append to existing person   (side-face match)
    │               → OR create new visitor profile (truly unknown)
```

---

## 🛠️ Development

### Running Tests (coming soon)
```bash
pip install pytest
pytest tests/
```

### Adding a New Model

1. Place the `.pt` file in `data/models/`
2. In the **Settings** dialog, select the new model from the dropdown
3. The detector reloads automatically — no restart needed

### Extending the Database Schema

The schema lives in `database.py → DatabaseManager.SCHEMA_SQL`. The database uses **SQLite WAL mode** with thread-local connections — each thread gets its own `Connection` object to avoid locking.

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## ❓ Troubleshooting

<details>
<summary><strong>Camera won't open</strong></summary>

- Ensure no other application is using the camera
- Try changing `CAMERA_INDEX` to `1` or `2` in Settings
- **macOS**: Grant camera permission in *System Settings → Privacy & Security → Camera*
- **Linux**: Add your user to the `video` group: `sudo usermod -aG video $USER`
</details>

<details>
<summary><strong>Recognition accuracy is low</strong></summary>

- Register with more photos at varied angles (8–15 recommended)
- Ensure training photos are well-lit and the face is clearly visible
- Lower `RECOGNITION_THRESHOLD` to `0.55` in Settings
- Clear all data and re-register if the initial registration was poor quality
</details>

<details>
<summary><strong>Model download fails on first run</strong></summary>

- Check your internet connection
- Manually download `yolo11n-pose.pt` from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in `data/models/`
- Manually download `buffalo_sc.zip` from [InsightFace](https://github.com/deepinsight/insightface), extract `w600k_mbf.onnx`, and place it in `data/models/`
</details>

<details>
<summary><strong>Low FPS / High CPU usage</strong></summary>

- Increase `SKIP_FRAMES` to `4` in Settings
- Reduce `DETECT_IMG_SIZE` to `320`
- Reduce camera resolution: set `CAMERA_WIDTH=320, CAMERA_HEIGHT=240`
- Switch to a smaller model (yolo11n-pose is the fastest)
</details>

<details>
<summary><strong>Raspberry Pi specific setup</strong></summary>

```bash
sudo apt install python3-tk libsdl2-mixer-2.0-0
pip install -r requirements.txt
```
Use `DETECT_IMG_SIZE=320` and `SKIP_FRAMES=4` for smooth performance on Pi 4.
</details>
