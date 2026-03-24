"""
config.py — 系統設定檔
所有路徑、閾值、硬體參數集中於此，修改設定只需改動本檔案。
"""

from dataclasses import dataclass, field
from pathlib import Path
import platform
import sys

# ──────────────────────────────────────────────
# 基礎路徑
# ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
MODEL_DIR      = DATA_DIR / "models"
DB_DIR         = DATA_DIR / "db"
SCREENSHOT_DIR = DATA_DIR / "screenshots"
LOG_DIR        = DATA_DIR / "logs"

# 首次執行時自動建立這些資料夾
for _d in [MODEL_DIR, DB_DIR, SCREENSHOT_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def _is_raspberry_pi() -> bool:
    machine = platform.machine().lower()
    return machine in ("armv7l", "aarch64", "armv6l")


# NOTE: 模型大小选项映射，允许用户在设定中选择不同大小的 YOLO-Pose 模型
YOLO_MODEL_OPTIONS = {
    "Nano (最快)": "yolo11n-pose.pt",
    "Small (平衡)": "yolo11s-pose.pt",
    "Medium (精准)": "yolo11m-pose.pt",
    "Large (最精准)": "yolo11l-pose.pt",
}


# ──────────────────────────────────────────────
# 主設定 dataclass
# ──────────────────────────────────────────────
@dataclass
class Config:
    # ── 攝影機 ──────────────────────────────
    CAMERA_INDEX:   int   = 0
    CAMERA_WIDTH:   int   = 640
    CAMERA_HEIGHT:  int   = 480
    CAMERA_FPS:     int   = 30

    # ── YOLOv11 檢測 ─────────────────────────
    YOLO_MODEL_NAME:     str   = "yolo11n-pose.pt"
    YOLO_MODEL_PATH:     Path  = field(default_factory=lambda: MODEL_DIR / "yolo11n-pose.pt")
    # NOTE: 0.45 召回更多遠距離/側面人體
    DETECT_CONFIDENCE:   float = 0.45
    DETECT_IOU:          float = 0.50
    DETECT_IMG_SIZE:     int   = 640
    DETECT_DEVICE:       str   = "cpu"
    MIN_FACE_SIZE:       int   = 40
    DETECT_CLASSES:      list  = field(default_factory=lambda: [0])

    # ── 特徵提取 (MobileFaceNet ONNX) ────────
    ONNX_MODEL_PATH:     Path  = field(default_factory=lambda: MODEL_DIR / "w600k_mbf.onnx")
    ONNX_MODEL_URL:      str   = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

    # ── 辨識閾值 ───────────────────────────
    RECOGNITION_THRESHOLD: float = 0.64
    # NOTE: 50 個向量支援全方位（正臉/側臉/俯仰）的特徵向量
    MAX_EMBEDDINGS_PER_PERSON: int = 50

    # ── 警報 ──────────────────────────────
    ALERT_SOUND_PATH:    Path  = field(default_factory=lambda: DATA_DIR / "alert.wav")
    ENABLE_SOUND:        bool  = True
    ALERT_COOLDOWN_S:    float = 5.0
    ALERT_FLASH_MS:      int   = 2000

    # ── GUI ───────────────────────────────
    WINDOW_TITLE:        str   = "YOLOv11 即時人臉辨識系統"
    VIDEO_DISPLAY_WIDTH: int   = 720
    VIDEO_DISPLAY_HEIGHT: int  = 540
    GUI_UPDATE_MS:       int   = 33
    SIDEBAR_WIDTH:       int   = 260

    # ── 路徑 ──────────────────────────────
    SCREENSHOT_DIR:      Path  = field(default_factory=lambda: SCREENSHOT_DIR)

    # 人臉框顏色 (OpenCV BGR)
    COLOR_KNOWN_BGR:    tuple  = (0, 200, 0)
    COLOR_UNKNOWN_BGR:  tuple  = (0, 165, 255)
    COLOR_ALERT_BGR:    tuple  = (0, 0, 255)
    BBOX_THICKNESS_NORMAL: int = 2
    BBOX_THICKNESS_ALERT:  int = 3

    # ── 效能 ──────────────────────────────
    # NOTE: SKIP_FRAMES=2 提高 CPU 利用率，增大隊列避免丟幀
    SKIP_FRAMES:         int   = 2
    FPS_WINDOW:          int   = 30
    FRAME_QUEUE_SIZE:    int   = 4
    DETECT_QUEUE_SIZE:   int   = 6
    RESULT_QUEUE_SIZE:   int   = 12
    DB_WRITE_QUEUE_SIZE: int   = 64

    # ── 資料庫 ─────────────────────────────
    DB_PATH:             Path  = field(default_factory=lambda: DB_DIR / "faces.db")

    # ── 批量導入 ───────────────────────────
    BATCH_LAPLACIAN_THRESHOLD: float = 50.0

    def __post_init__(self):
        if _is_raspberry_pi():
            if self.DETECT_IMG_SIZE > 320:
                self.DETECT_IMG_SIZE = 320
            if self.SKIP_FRAMES < 4:
                self.SKIP_FRAMES = 4
            if self.CAMERA_FPS > 15:
                self.CAMERA_FPS = 15
