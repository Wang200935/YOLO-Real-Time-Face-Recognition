"""
utils.py — 通用工具函式
包含：FPS 計數器、下載工具、警報管理器、影像轉換工具
"""

import time
import threading
import logging
import zipfile
import shutil
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# FPS 計數器（ring buffer 平均）
# ──────────────────────────────────────────────
class FPSCounter:
    """使用滑動視窗計算即時 FPS。"""

    def __init__(self, window: int = 30):
        self._timestamps: deque = deque(maxlen=window)

    def tick(self) -> float:
        """每處理一幀呼叫一次，回傳當前 FPS。"""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def get(self) -> float:
        """取得最新 FPS（不更新時間戳）。"""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self):
        self._timestamps.clear()


# ──────────────────────────────────────────────
# 警報管理器
# ──────────────────────────────────────────────
class AlertManager:
    """管理警報冷卻時間與音效播放。"""

    def __init__(self, cooldown_s: float, sound_path: Path, enable_sound: bool = True):
        self._cooldown_s  = cooldown_s
        self._sound_path  = sound_path
        self._enable_sound = enable_sound
        self._last_alert: dict[str, float] = {}  # name -> timestamp
        self._lock = threading.Lock()
        self._sound_obj = None
        self._mixer_ready = False
        if enable_sound:
            self._init_mixer()

    def _init_mixer(self):
        """初始化 pygame.mixer，失敗則靜默模式。"""
        try:
            import pygame
            pygame.mixer.init()
            self._mixer_ready = True
            logger.info("pygame.mixer 初始化成功")
        except Exception as e:
            logger.warning(f"音效初始化失敗，靜默模式: {e}")
            self._mixer_ready = False

    def trigger(self, name: str) -> bool:
        """
        嘗試觸發警報。
        若距離上次警報未超過冷卻時間則跳過。
        回傳 True 表示警報已觸發。
        """
        with self._lock:
            now = time.time()
            last = self._last_alert.get(name, 0.0)
            if now - last < self._cooldown_s:
                return False
            self._last_alert[name] = now

        if self._enable_sound and self._mixer_ready:
            threading.Thread(target=self._play_sound, daemon=True).start()
        return True

    def _play_sound(self):
        """非阻塞播放警報音效。"""
        try:
            import pygame
            if self._sound_path.exists():
                sound = pygame.mixer.Sound(str(self._sound_path))
                sound.play()
                time.sleep(sound.get_length())
            else:
                # 若音效檔不存在，使用系統提示音（macOS）
                import subprocess, platform
                if platform.system() == "Darwin":
                    subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"],
                                   capture_output=True)
        except Exception as e:
            logger.debug(f"音效播放失敗: {e}")

    def clear_cooldown(self, name: str):
        """清除特定人物的冷卻計時。"""
        with self._lock:
            self._last_alert.pop(name, None)


# ──────────────────────────────────────────────
# 檔案下載工具
# ──────────────────────────────────────────────
def download_file(
    url: str,
    dest: Path,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> None:
    """
    串流下載檔案至 dest，支援進度回呼。
    progress_callback(downloaded_bytes, total_bytes, filename)
    失敗時拋出 RuntimeError。
    """
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536  # 64 KB

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total, dest.name)

        tmp.rename(dest)
        logger.info(f"下載完成: {dest.name}")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"下載失敗 {url}: {e}") from e


def extract_zip_member(zip_path: Path, member: str, dest: Path) -> bool:
    """
    從 zip 檔案中提取特定成員至 dest。
    回傳 True 表示成功。
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 尋找符合 member 名稱的檔案（忽略路徑）
            for name in zf.namelist():
                if Path(name).name == member:
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info(f"已提取: {member} → {dest}")
                    return True
        logger.warning(f"在 {zip_path.name} 中找不到 {member}")
        return False
    except Exception as e:
        logger.error(f"ZIP 提取失敗: {e}")
        return False


# ──────────────────────────────────────────────
# 影像工具
# ──────────────────────────────────────────────
def frame_to_photoimage(frame: np.ndarray, width: int, height: int) -> ImageTk.PhotoImage:
    """
    BGR numpy 陣列 → 縮放後的 Tkinter PhotoImage。
    保持比例縮放至指定寬高範圍內。
    """
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """BGR numpy → JPEG bytes，用於縮圖儲存。"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf)


def decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """JPEG bytes → BGR numpy，失敗回傳 None。"""
    try:
        buf = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def laplacian_variance(img: np.ndarray) -> float:
    """計算影像的拉普拉斯方差，用於模糊度判斷。值越大越清晰。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_face(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              margin: float = 0.1) -> np.ndarray:
    """
    從幀中裁切人臉 ROI，可加入邊距比例。
    座標自動 clip 至影像邊界。
    """
    h, w = frame.shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)
    return frame[y1:y2, x1:x2].copy()


def draw_detection(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    name: str,
    confidence: float,
    color: tuple,
    thickness: int = 2,
) -> np.ndarray:
    """在幀上繪製人臉框與標籤（不修改原始幀，回傳副本）。"""
    out = frame  # 在外部已做 copy，直接繪製
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

    label = f"{name} {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(y1 - 5, th + baseline)
    cv2.rectangle(out, (x1, ty - th - baseline - 2), (x1 + tw, ty), color, -1)
    cv2.putText(out, label, (x1, ty - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────
# 執行緒安全計時器
# ──────────────────────────────────────────────
class PerformanceTimer:
    """測量函式執行耗時（毫秒），線程安全。"""

    def __init__(self, name: str, window: int = 20):
        self._name = name
        self._samples: deque = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, ms: float):
        with self._lock:
            self._samples.append(ms)

    def avg_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    def __repr__(self):
        return f"{self._name}: {self.avg_ms():.1f}ms"
