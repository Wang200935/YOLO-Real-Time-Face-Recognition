"""
gui.py — Tkinter 主視窗
布局：左側 Canvas 影像（自動縮放）+ 右側固定 260px 側欄
"""

import logging
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from face_detector import SKELETON_CONNECTIONS, _limb_color
from utils import FPSCounter, encode_jpeg, laplacian_variance

logger = logging.getLogger(__name__)


# ── CJK 文字渲染工具 ─────────────────────────────
def _load_cjk_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """載入支援中文的字體，依平台嘗試常見字體路徑。"""
    import platform
    candidates = []
    system = platform.system()
    if system == "Darwin":
        candidates = [
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    elif system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


# 全局字體快取
_LABEL_FONT: Optional[ImageFont.FreeTypeFont] = None


def _get_label_font() -> ImageFont.FreeTypeFont:
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = _load_cjk_font(16)
    return _LABEL_FONT


def draw_labels_pil(frame: np.ndarray,
                    labels: list[tuple[str, int, int, tuple]]):
    """
    一次性在 frame 上繪製所有中文標籤（只做一次 BGR↔RGB 轉換）。
    labels: [(text, x, y, bg_color_bgr), ...]
    """
    if not labels:
        return
    font = _get_label_font()
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, x, y, bg_color in labels:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        ty = max(y - th - 6, 0)
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle([x, ty, x + tw + 6, ty + th + 4], fill=bg_rgb)
        draw.text((x + 3, ty + 1), text, font=font, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class MainWindow:
    """
    Tkinter 主視窗。
    所有 widget 更新必須在主執行緒執行（透過 root.after）。
    """

    _AUTO_ACCUMULATE_INTERVAL = 1.0    # NOTE: 1 秒間隔，更頻繁累積全方位向量
    _MIN_QUALITY = 60.0                # NOTE: 側臉清晰度天然低，放寬品質閾值
    _BUFFER_COLLECT_MAX = 8            # NOTE: 8 幀上限，中距離/遠距離更容易收滿
    _BUFFER_KEEP_TOP = 6               # 註冊時只保留品質最高的 N 幀
    _BUFFER_MIN_SAMPLES = 3            # NOTE: 3 幀即可註冊，對全身/遠距離場景更友好
    _BUFFER_TIMEOUT = 10.0             # 緩衝區超時（秒）
    # NOTE: 0.45 是合理的一致性閾值
    _BUFFER_SIM_THRESHOLD = 0.45       # 緩衝區內部一致性閾值
    _BUFFER_QUALITY_FLOOR = 50.0       # NOTE: 緩衝期品質閾值降低，全身場景品質更低
    _MIN_FACE_PIXELS = 80              # NOTE: 降低最小像素門檻，支持遠距離小臉

    def __init__(self, config, recognizer, db_manager,
                 result_queue: queue.Queue,
                 thread_manager,
                 alert_manager,
                 alert_names: Optional[set] = None,
                 alert_lock: Optional[threading.Lock] = None):
        self._config        = config
        self._recognizer    = recognizer
        self._db            = db_manager
        self._result_q      = result_queue
        self._thread_mgr    = thread_manager
        self._alert_mgr     = alert_manager

        self._fps_counter   = FPSCounter(window=config.FPS_WINDOW)
        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self._is_running    = False

        self._alert_names: set[str] = alert_names if alert_names is not None else set()
        self._alert_lock: threading.Lock = alert_lock if alert_lock is not None else threading.Lock()

        # 累積特徵計時器：{person_id: last_accumulate_time}
        self._accumulate_timers: dict[int, float] = {}

        # 未知人臉緩衝區：收集多幀後才註冊
        self._unknown_buffer: dict = self._empty_buffer()
        self._delete_cooldown_until: float = 0.0  # 刪除後暫停自動註冊
        self._restarting: bool = False
        self._canvas_w: int = config.VIDEO_DISPLAY_WIDTH
        self._canvas_h: int = config.VIDEO_DISPLAY_HEIGHT

        self._build_root()
        self._build_styles()
        self._build_layout()
        self._refresh_person_list()

    def _build_root(self):
        self.root = tk.Tk()
        self.root.title(self._config.WINDOW_TITLE)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        min_w = self._config.VIDEO_DISPLAY_WIDTH + self._config.SIDEBAR_WIDTH + 30
        min_h = self._config.VIDEO_DISPLAY_HEIGHT + 100
        self.root.minsize(min_w, min_h)

    def _build_styles(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Title.TLabel",   font=("Helvetica", 14, "bold"))
        s.configure("Section.TLabel", font=("Helvetica", 11, "bold"))
        s.configure("Info.TLabel",    font=("Helvetica", 9))
        s.configure("Action.TButton", font=("Helvetica", 9), padding=4)

    def _build_layout(self):
        C = self._config

        # ── Toolbar ─────────────────────────────────
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 0))

        ttk.Label(toolbar, text="YOLOv11 即時人臉辨識系統",
                  style="Title.TLabel").pack(side=tk.LEFT)

        self._btn_toggle = ttk.Button(
            toolbar, text="啟動辨識",
            command=self._toggle_recognition,
            style="Action.TButton")
        self._btn_toggle.pack(side=tk.RIGHT, padx=(4, 0))

        # 攝影機選擇
        ttk.Label(toolbar, text="攝影機:", style="Info.TLabel").pack(side=tk.RIGHT, padx=(8, 2))
        self._camera_var = tk.IntVar(value=C.CAMERA_INDEX)
        cam_spin = ttk.Spinbox(toolbar, from_=0, to=9, width=3,
                                textvariable=self._camera_var,
                                command=self._on_camera_change)
        cam_spin.pack(side=tk.RIGHT)

        # ── 主內容區 ────────────────────────────────
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # 左側影像 Canvas（自動縮放）
        video_frame = ttk.Frame(content)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._video_canvas = tk.Canvas(
            video_frame,
            highlightthickness=0,
        )
        self._video_canvas.pack(fill=tk.BOTH, expand=True)
        self._video_canvas.bind("<Configure>", self._on_canvas_resize)

        # 佔位符
        self._placeholder_id = self._video_canvas.create_text(
            C.VIDEO_DISPLAY_WIDTH // 2,
            C.VIDEO_DISPLAY_HEIGHT // 2,
            text="按下「啟動辨識」開始",
            font=("Helvetica", 18),
            tags="placeholder",
        )

        # 右側固定寬度側欄
        sidebar = ttk.Frame(content, width=C.SIDEBAR_WIDTH)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        self._build_sidebar(sidebar)

        # ── 警報橫幅（平時隱藏）──────────────────────
        self._alert_banner = tk.Label(
            self.root, text="",
            font=("Helvetica", 13, "bold"),
            fg="white", bg="#cc0000",
            pady=6,
        )

        # ── 狀態列 ────────────────────────────────────
        self._status_frame = ttk.Frame(self.root)
        self._status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self._lbl_status = ttk.Label(
            self._status_frame,
            text="就緒 — 等待啟動",
            style="Info.TLabel",
            anchor=tk.W,
        )
        self._lbl_status.pack(fill=tk.X, padx=10, pady=6)

    def _build_sidebar(self, parent):
        ttk.Label(parent, text="已辨識人員",
                  style="Section.TLabel").pack(anchor=tk.W, padx=8, pady=(8, 2))

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._person_listbox = tk.Listbox(
            list_frame,
            activestyle="none",
            yscrollcommand=scrollbar.set,
            height=12,
        )
        self._person_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self._person_listbox.yview)

        self._person_listbox.bind("<Double-Button-1>", self._on_rename)
        self._person_listbox.bind("<Button-2>", self._show_context_menu)
        self._person_listbox.bind("<Button-3>", self._show_context_menu)

        self._lbl_count = ttk.Label(parent, text="共 0 人", style="Info.TLabel")
        self._lbl_count.pack(anchor=tk.W, padx=8, pady=(2, 4))

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)

        buttons = [
            ("新增人物", self._open_add_person),
            ("批量導入", self._open_batch_import),
            ("切換警示", self._toggle_alert),
            ("重新命名", self._on_rename_btn),
            ("刪除人物", self._on_delete),
            ("全部刪除", self._on_delete_all),
        ]
        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd,
                       style="Action.TButton").pack(fill=tk.X, pady=2)

        # 設定按鈕
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(parent, text="設定", command=self._open_settings,
                   style="Action.TButton").pack(fill=tk.X, padx=8, pady=2)

        # 識別閾值
        ttk.Label(parent, text="識別閾值:",
                  style="Info.TLabel").pack(anchor=tk.W, padx=8, pady=(8, 0))

        self._threshold_var = tk.DoubleVar(value=self._config.RECOGNITION_THRESHOLD)
        ttk.Scale(parent, from_=0.20, to=0.85,
                  variable=self._threshold_var,
                  orient=tk.HORIZONTAL,
                  command=self._on_threshold_change).pack(fill=tk.X, padx=8)

        self._lbl_threshold = ttk.Label(
            parent,
            text=f"{self._config.RECOGNITION_THRESHOLD:.2f}",
            style="Info.TLabel",
        )
        self._lbl_threshold.pack(anchor=tk.W, padx=8, pady=(0, 8))

    def _on_canvas_resize(self, event):
        """Canvas 大小改變時更新記錄的尺寸。"""
        self._canvas_w = event.width
        self._canvas_h = event.height
        # 更新佔位符位置
        if self._placeholder_id:
            self._video_canvas.coords(
                "placeholder", event.width // 2, event.height // 2)

    # ── 主事件循環 ────────────────────────────────────
    def run(self):
        self._schedule_update()
        self.root.mainloop()

    def _schedule_update(self):
        self.root.after(self._config.GUI_UPDATE_MS, self._update_frame)

    def _update_frame(self):
        try:
            latest = None
            count = 0
            while True:
                try:
                    item = self._result_q.get_nowait()
                    count += 1
                    if item is not None:
                        latest = item
                except queue.Empty:
                    break

            if latest is not None:
                from worker_threads import FrameResult
                frame_result: FrameResult = latest
                self._display_frame(frame_result)
                self._fps_counter.tick()
                self._check_alerts(frame_result)
                self._auto_register_unknown(frame_result)
            elif self._is_running and not hasattr(self, '_frame_warned'):
                # 首次 10 秒無畫面時輸出警告
                if not hasattr(self, '_start_wait_time'):
                    self._start_wait_time = time.time()
                elif time.time() - self._start_wait_time > 10:
                    logger.warning("[GUI] 已等待 10 秒仍未收到影像幀，請檢查攝影機與模型載入狀態")
                    self._frame_warned = True

            self._update_status_bar()
        except Exception as e:
            logger.error(f"[GUI] update_frame 異常: {e}", exc_info=True)
        finally:
            self._schedule_update()

    def _display_frame(self, frame_result):
        frame = frame_result.frame.copy()

        # 收集所有標籤，一次性 PIL 渲染
        labels = []
        for res in frame_result.results:
            if res.is_alert:
                color = self._config.COLOR_ALERT_BGR
                thickness = self._config.BBOX_THICKNESS_ALERT
            elif res.is_unknown:
                color = self._config.COLOR_UNKNOWN_BGR
                thickness = self._config.BBOX_THICKNESS_NORMAL
            else:
                color = self._config.COLOR_KNOWN_BGR
                thickness = self._config.BBOX_THICKNESS_NORMAL

            # 繪製人體 bbox
            cv2.rectangle(frame, (res.x1, res.y1), (res.x2, res.y2), color, thickness)

            # 繪製人臉區域框（虛線效果用短線段模擬）
            if res.face_x2 > res.face_x1 and res.face_y2 > res.face_y1:
                face_color = (255, 255, 0)  # 青色 BGR
                cv2.rectangle(frame, (res.face_x1, res.face_y1),
                              (res.face_x2, res.face_y2), face_color, 1)

            # 繪製骨架
            if res.body_keypoints is not None:
                kpts = res.body_keypoints  # (17, 3) = [x, y, conf]
                # 繪製骨架連線
                for i, j in SKELETON_CONNECTIONS:
                    if kpts[i, 2] >= 0.3 and kpts[j, 2] >= 0.3:
                        pt1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                        pt2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        limb_color = _limb_color(i, j)
                        cv2.line(frame, pt1, pt2, limb_color, 2, cv2.LINE_AA)
                # 繪製關鍵點
                for k in range(17):
                    if kpts[k, 2] >= 0.3:
                        cx, cy = int(kpts[k, 0]), int(kpts[k, 1])
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

            # 只對有辨識結果的人顯示標籤，"--" 表示偵測到人但沒看到臉
            if res.name != "--":
                label = f"{res.name} {res.confidence:.2f}"
                labels.append((label, res.x1, res.y1, color))

        # 批次繪製所有中文標籤（只做一次 BGR↔PIL 轉換）
        draw_labels_pil(frame, labels)

        # 自動縮放至 Canvas 尺寸
        h, w = frame.shape[:2]
        cw = max(self._canvas_w, 1)
        ch = max(self._canvas_h, 1)
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw < 1 or nh < 1:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._current_photo = photo

        self._video_canvas.delete("video_frame")
        self._video_canvas.create_image(cw // 2, ch // 2,
                                         anchor=tk.CENTER,
                                         image=photo,
                                         tags="video_frame")

        if self._placeholder_id:
            self._video_canvas.delete("placeholder")
            self._placeholder_id = None

    def _check_alerts(self, frame_result):
        alert_triggered = []
        for res in frame_result.results:
            if res.is_alert:
                fired = self._alert_mgr.trigger(res.name)
                if fired:
                    alert_triggered.append(res.name)
        if alert_triggered:
            names_str = ", ".join(alert_triggered)
            self._show_alert_banner(f"警示！偵測到: {names_str}")

    @staticmethod
    def _empty_buffer() -> dict:
        return {"embeddings": [], "crops": [], "qualities": [],
                "first_seen": 0.0, "best_quality": 0.0, "best_crop": None}

    def _auto_register_unknown(self, frame_result):
        """
        自動處理辨識結果：
        1. 已知人臉 → 定期累積特徵向量（增量學習）
        2. 未知人臉 → 加入緩衝區 → 收集 N 幀後一次性註冊
        """
        now = time.time()

        # 刪除後冷卻期：避免立即重新註冊
        if now < self._delete_cooldown_until:
            return

        for res in frame_result.results:
            if res.embedding is None or res.name == "--":
                continue

            # ── 已知人臉：定期累積新角度的特徵 ──
            if not res.is_unknown and res.person_id is not None:
                self._maybe_accumulate(res, frame_result.frame, now)
                continue

            if not res.is_unknown:
                continue

            # ── 未知人臉：品質與大小過濾 ──
            # 使用人臉區域尺寸（若有），否則退回人體 bbox
            if res.face_x2 > res.face_x1 and res.face_y2 > res.face_y1:
                face_w = res.face_x2 - res.face_x1
                face_h = res.face_y2 - res.face_y1
            else:
                face_w = res.x2 - res.x1
                face_h = res.y2 - res.y1
            if face_w < self._MIN_FACE_PIXELS or face_h < self._MIN_FACE_PIXELS:
                continue  # 臉太小，截圖一定糊
            if res.quality < self._MIN_QUALITY:
                continue  # 太模糊

            # 先嘗試寬鬆匹配已知人物
            matched_id = self._try_match_existing(res.embedding)
            if matched_id is not None:
                self._maybe_add_to_existing(matched_id, res, frame_result.frame, now)
                continue

            # 加入未知緩衝區
            self._buffer_unknown(res, frame_result.frame, now)
            break

        # 檢查緩衝區是否超時
        buf = self._unknown_buffer
        if buf["first_seen"] > 0 and now - buf["first_seen"] > self._BUFFER_TIMEOUT:
            if len(buf["embeddings"]) >= self._BUFFER_MIN_SAMPLES:
                self._flush_buffer()
            else:
                logger.debug(f"[GUI] 緩衝區超時丟棄 ({len(buf['embeddings'])} 幀不足)")
                self._restore_skip_frames()
                self._unknown_buffer = self._empty_buffer()

    def _buffer_unknown(self, res, frame: np.ndarray, now: float):
        """將未知人臉加入緩衝區，多收集後篩選品質最高的幀進行註冊。"""
        buf = self._unknown_buffer
        emb = res.embedding.copy()

        # 緩衝區已有資料，檢查是否是同一個人
        if buf["embeddings"]:
            avg_emb = np.mean(buf["embeddings"], axis=0)
            avg_emb = avg_emb / max(np.linalg.norm(avg_emb), 1e-8)
            sim = float(emb @ avg_emb)
            if sim < self._BUFFER_SIM_THRESHOLD:
                # 不是同一個人，重置緩衝區
                logger.debug(f"[GUI] 緩衝區重置: 新臉與緩衝不符 (sim={sim:.3f})")
                self._restore_skip_frames()
                self._unknown_buffer = self._empty_buffer()
                buf = self._unknown_buffer

        # NOTE: 緩衝開始時臨時提高幀率（SKIP_FRAMES=1）以獲取更多高品質幀
        if not buf["embeddings"]:
            buf["first_seen"] = now
            self._boost_frame_rate()

        # 緩衝期間用更寬鬆的品質閾值，多收集後節篩選
        if res.quality < self._BUFFER_QUALITY_FLOOR:
            return  # 太模糊，連緩衝都不收

        buf["embeddings"].append(emb)
        buf["qualities"].append(res.quality)

        # 從原始幀切較大區域作為截圖
        face_crop = self._crop_face_from_frame(frame, res)

        # 保留最清楚的那張截圖
        if res.quality > buf["best_quality"]:
            buf["best_quality"] = res.quality
            buf["best_crop"] = face_crop

        collected = len(buf['embeddings'])
        logger.info(f"[GUI] 緩衝未知人臉: {collected}/{self._BUFFER_COLLECT_MAX} "
                     f"(清晰度={res.quality:.0f})")

        # 收集到足夠樣本，篩選後註冊
        if collected >= self._BUFFER_COLLECT_MAX:
            self._flush_buffer()
        # NOTE: 提前註冊 — 已過 3 秒且有 3+ 幀，不等滿上限直接註冊
        # 對中距離/遠距離場景更友好，避免等待太久
        elif collected >= self._BUFFER_MIN_SAMPLES and now - buf["first_seen"] > 3.0:
            logger.info(f"[GUI] 提前註冊: {collected} 幀已足夠且超過 3 秒")
            self._flush_buffer()

    def _crop_face_from_frame(self, frame: np.ndarray, res) -> np.ndarray:
        """從原始幀裁切人臉區域（加 50% 邊距），不是 112x112 對齊臉。
        優先使用骨架衍生的人臉區域，否則退回全身 bbox。
        NOTE: 邊距從 30% 擴大至 50%，裁切更大上下文提升畫質。"""
        h, w = frame.shape[:2]
        # 優先使用人臉區域
        if res.face_x2 > res.face_x1 and res.face_y2 > res.face_y1:
            fx1, fy1, fx2, fy2 = res.face_x1, res.face_y1, res.face_x2, res.face_y2
        else:
            fx1, fy1, fx2, fy2 = res.x1, res.y1, res.x2, res.y2
        fw, fh = fx2 - fx1, fy2 - fy1
        # NOTE: 邊距從 30% 擴大至 50%，裁切更大上下文
        mx, my = int(fw * 0.5), int(fh * 0.5)
        x1 = max(0, fx1 - mx)
        y1 = max(0, fy1 - my)
        x2 = min(w, fx2 + mx)
        y2 = min(h, fy2 + my)
        crop = frame[y1:y2, x1:x2].copy()
        # NOTE: 裁切後如果尺寸太小，上採樣至最小 200x200 保障畫質
        if crop.size > 0:
            ch, cw = crop.shape[:2]
            if ch < 200 or cw < 200:
                scale = max(200 / cw, 200 / ch)
                new_w = int(cw * scale)
                new_h = int(ch * scale)
                crop = cv2.resize(crop, (new_w, new_h),
                                  interpolation=cv2.INTER_CUBIC)
        return crop

    def _flush_buffer(self):
        """將緩衝區的多幀資料篩選后一次性註冊為新人物。"""
        buf = self._unknown_buffer
        self._unknown_buffer = self._empty_buffer()
        self._restore_skip_frames()

        embeddings = buf["embeddings"]
        qualities = buf["qualities"]
        best_crop = buf["best_crop"]

        if not embeddings:
            return

        # NOTE: 按品質排序，只保留 top-N 品質最高的幀
        paired = list(zip(qualities, embeddings))
        paired.sort(key=lambda x: x[0], reverse=True)
        top_pairs = paired[:self._BUFFER_KEEP_TOP]

        # 過濾品質低於 _MIN_QUALITY 的幀
        filtered = [(q, e) for q, e in top_pairs if q >= self._MIN_QUALITY]

        if len(filtered) < 3:
            logger.info(f"[GUI] 緩衝篩選後只剩 {len(filtered)} 幀（不足 3），放棄註冊")
            return

        selected_embeddings = [e for _, e in filtered]
        logger.info(f"[GUI] 緩衝篩選: {len(embeddings)} 幀 → {len(selected_embeddings)} 幀 "
                     f"(品質範圍 {filtered[-1][0]:.0f}~{filtered[0][0]:.0f})")

        threading.Thread(
            target=self._do_auto_register_batch,
            args=(selected_embeddings, best_crop),
            daemon=True,
        ).start()

    def _boost_frame_rate(self):
        """NOTE: 緩衝開始時臨時設 SKIP_FRAMES=1 提高幀率以收集更清晰的幀。"""
        if not hasattr(self, '_original_skip_frames'):
            self._original_skip_frames = self._config.SKIP_FRAMES
            self._config.SKIP_FRAMES = 1
            logger.debug(f"[GUI] 緩衝幀率提升: SKIP_FRAMES {self._original_skip_frames} → 1")

    def _restore_skip_frames(self):
        """恢復原始的 SKIP_FRAMES 設定。"""
        if hasattr(self, '_original_skip_frames'):
            self._config.SKIP_FRAMES = self._original_skip_frames
            logger.debug(f"[GUI] 幀率恢復: SKIP_FRAMES → {self._original_skip_frames}")
            del self._original_skip_frames

    def _try_match_existing(self, embedding: np.ndarray) -> Optional[int]:
        """用寬鬆閾值嘗試匹配已知人物（側臉相似度更低，需要更寬鬆）。"""
        try:
            # NOTE: 係數 0.35 — 側臉與正臉 embedding 差異大
            # RECOGNITION_THRESHOLD=0.64 × 0.35 ≈ 0.22，極低閾值防止側臉被當新人
            loose_threshold = self._config.RECOGNITION_THRESHOLD * 0.35
            best_id, best_name, best_sim = self._recognizer.best_match(embedding)
            if best_id is not None and best_sim >= loose_threshold:
                return best_id
        except Exception:
            pass
        return None

    def _maybe_add_to_existing(self, person_id: int, res, frame, now: float):
        """寬鬆匹配到已知人物，追加特徵。"""
        last_t = self._accumulate_timers.get(person_id, 0.0)
        if now - last_t < self._AUTO_ACCUMULATE_INTERVAL:
            return
        self._accumulate_timers[person_id] = now
        emb = res.embedding.copy()
        crop = self._crop_face_from_frame(frame, res)
        threading.Thread(
            target=self._do_add_embedding_to_existing,
            args=(person_id, emb, crop),
            daemon=True,
        ).start()

    _AUTO_ACCUMULATE_INTERVAL = 1.0    # NOTE: 1秒間隔，更頻繁累積向量覆蓋更多角度

    def _maybe_accumulate(self, res, frame: np.ndarray, now: float):
        """
        已知人臉定期累積高品質特徵（增量學習）。
        NOTE: 側臉/俯仰清晰度天然低於正臉，降低品質要求以收集全方位向量。
        同時全身可見時額外截全身照。
        """
        pid = res.person_id
        last_t = self._accumulate_timers.get(pid, 0.0)
        if now - last_t < self._AUTO_ACCUMULATE_INTERVAL:
            return
        # NOTE: 側臉/俯仰清晰度較低，用更寬鬆的品質閾值（60）
        if res.quality < 60.0:
            return
        self._accumulate_timers[pid] = now
        emb = res.embedding.copy()
        face_crop = self._crop_face_from_frame(frame, res)

        # NOTE: 全身可見時額外截全身照（強化全身場景辨識）
        body_crop = None
        body_h = res.y2 - res.y1
        face_h = res.face_y2 - res.face_y1
        if body_h > face_h * 2.5 and body_h > 100:
            h, w = frame.shape[:2]
            bx1 = max(0, res.x1 - 10)
            by1 = max(0, res.y1 - 10)
            bx2 = min(w, res.x2 + 10)
            by2 = min(h, res.y2 + 10)
            body_crop = frame[by1:by2, bx1:bx2].copy()

        threading.Thread(
            target=self._do_add_embedding_to_existing,
            args=(pid, emb, face_crop, body_crop),
            daemon=True,
        ).start()

    def _do_add_embedding_to_existing(self, person_id: int,
                                       embedding: np.ndarray,
                                       face_crop: Optional[np.ndarray] = None,
                                       body_crop: Optional[np.ndarray] = None):
        """追加特徵向量到已存在的人物，同時保存臉部和全身截圖。"""
        try:
            self._recognizer.add_face(person_id, embedding)
            name = self._db.get_person_name(person_id) or f"id_{person_id}"
            save_dir = self._config.SCREENSHOT_DIR / name
            save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")

            if face_crop is not None and face_crop.size > 0:
                try:
                    cv2.imwrite(str(save_dir / f"{ts}_face.jpg"), face_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception:
                    pass

            # NOTE: 全身照存於同目錄，用 _body 後綴區分
            if body_crop is not None and body_crop.size > 0:
                try:
                    cv2.imwrite(str(save_dir / f"{ts}_body.jpg"), body_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"[GUI] 追加特徵失敗: {e}")

    def _do_auto_register_batch(self, embeddings: list[np.ndarray],
                                 best_crop: Optional[np.ndarray]):
        """用多個 embedding 一次性註冊新人物（更穩定的初始辨識）。
        NOTE: 註冊前會做最終復核 — 用平均向量與已有人物比對，
        防止側臉/俯仰角被誤註冊為新人。
        """
        try:
            # ── 最終復核：防止側臉被誤註冊為新人 ──
            avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
            avg_emb = avg_emb / max(np.linalg.norm(avg_emb), 1e-8)
            # NOTE: 超寬鬆閾值 0.20 — 只要有一點點相似就歸類到已知人物
            # 正臉 vs 側臉的 cosine sim 通常在 0.2~0.5 之間
            FINAL_CHECK_THRESHOLD = 0.20
            best_id, best_name, best_sim = self._recognizer.best_match(avg_emb)
            if best_id is not None and best_sim >= FINAL_CHECK_THRESHOLD:
                # 匹配成功：追加到已知人物而非建立新人
                logger.info(f"[GUI] 最終復核匹配: {best_name} (sim={best_sim:.3f})，"
                            f"追加 {len(embeddings)} 個向量而非新增")
                for emb in embeddings:
                    self._recognizer.add_face(best_id, emb)
                return

            # ── 未匹配到任何人 → 正常註冊新人 ──
            visitor_name = self._db.get_next_visitor_name()
            person_id = self._db.add_person(visitor_name)

            # 批量寫入所有特徵向量（一次 rebuild_index）
            self._recognizer.batch_add_faces(person_id, embeddings)

            # 儲存最清楚的截圖
            if best_crop is not None and best_crop.size > 0:
                try:
                    save_dir = self._config.SCREENSHOT_DIR / visitor_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / "best.jpg"), best_crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception as e:
                    logger.warning(f"[GUI] 儲存截圖失敗: {e}")
                try:
                    self._db.set_thumbnail(person_id, encode_jpeg(best_crop))
                except Exception:
                    pass

            logger.info(f"[GUI] 自動新增: {visitor_name} "
                        f"({len(embeddings)} 個特徵向量)")
            self.root.after(0, self._refresh_person_list)
        except Exception as e:
            logger.error(f"[GUI] 自動新增人物失敗: {e}")

    def _show_alert_banner(self, text: str):
        self._alert_banner.config(text=text)
        self._alert_banner.pack(fill=tk.X, before=self._status_frame)
        self.root.after(self._config.ALERT_FLASH_MS, self._hide_alert_banner)

    def _hide_alert_banner(self):
        self._alert_banner.pack_forget()

    def _update_status_bar(self):
        # 降低更新頻率：每 500ms 更新一次狀態列（避免頻繁 DB 查詢）
        now = time.time()
        if hasattr(self, '_last_status_update') and now - self._last_status_update < 0.5:
            return
        self._last_status_update = now
        fps = self._fps_counter.get()
        try:
            stats = self._db.get_stats()
        except Exception:
            stats = {"persons": 0, "embeddings": 0}
        detect_ms = self._thread_mgr.detect_ms if self._thread_mgr else 0
        recog_ms  = self._thread_mgr.recog_ms  if self._thread_mgr else 0
        state = "辨識中" if self._is_running else "已停止"
        text = (f"FPS: {fps:.1f} | 偵測: {detect_ms:.0f}ms | "
                f"辨識: {recog_ms:.0f}ms | "
                f"資料庫: {stats['persons']} 人 | {state}")
        self._lbl_status.config(text=text)

    # ── 辨識控制 ─────────────────────────────────────
    def _toggle_recognition(self):
        if self._is_running:
            self._stop_recognition()
        else:
            self._start_recognition()

    def _start_recognition(self):
        try:
            self._thread_mgr.start_all()
            self._is_running = True
            self._btn_toggle.config(text="停止辨識")
            self._lbl_status.config(text="啟動中...")
            logger.info("[GUI] 辨識已啟動")
            # 延遲檢查攝影機是否啟動成功
            self.root.after(2000, self._check_camera_status)
        except Exception as e:
            messagebox.showerror("啟動失敗", str(e))

    def _check_camera_status(self):
        """啟動 2 秒後檢查攝影機狀態。"""
        cam = self._thread_mgr.camera_thread
        if cam and cam.error:
            self._lbl_status.config(text=f"攝影機錯誤: {cam.error}")
            messagebox.showerror("攝影機錯誤", cam.error)
            self._stop_recognition()

    def _stop_recognition(self):
        """停止辨識（非同步，不阻塞 GUI）。"""
        if self._restarting:
            return
        self._is_running = False
        self._btn_toggle.config(text="停止中...", state=tk.DISABLED)
        # 重置等待警告
        for attr in ('_start_wait_time', '_frame_warned'):
            if hasattr(self, attr):
                delattr(self, attr)

        def _do_stop():
            self._thread_mgr.stop_all(timeout=3.0)
            self.root.after(0, self._finish_stop)

        threading.Thread(target=_do_stop, daemon=True).start()

    def _finish_stop(self):
        """stop_all 完成後在主執行緒更新 UI。"""
        self._restarting = False
        self._btn_toggle.config(text="啟動辨識", state=tk.NORMAL)
        self._lbl_status.config(text="已停止")
        self._video_canvas.delete("video_frame")
        cw = max(self._canvas_w, 1)
        ch = max(self._canvas_h, 1)
        self._placeholder_id = self._video_canvas.create_text(
            cw // 2, ch // 2,
            text="按下「啟動辨識」開始",
            font=("Helvetica", 18),
            tags="placeholder",
        )
        logger.info("[GUI] 辨識已停止")

    # ── 人物列表 ─────────────────────────────────────
    def _refresh_person_list(self):
        self._person_listbox.delete(0, tk.END)
        persons = self._db.get_all_persons()
        for p in persons:
            name = p["name"]
            with self._alert_lock:
                is_alert = name in self._alert_names
            prefix = "[!] " if is_alert else ""
            self._person_listbox.insert(tk.END, f"{prefix}{name}")
        self._lbl_count.config(text=f"共 {len(persons)} 人")

    def _get_selected_person(self) -> Optional[dict]:
        sel = self._person_listbox.curselection()
        if not sel:
            return None
        display = self._person_listbox.get(sel[0])
        if display.startswith("[!] "):
            name = display[4:]
        else:
            name = display.strip()
        return self._db.get_person_by_name(name)

    # ── 人物操作 ─────────────────────────────────────
    def _open_add_person(self):
        AddPersonDialog(self.root, self._config, self._recognizer,
                        self._db, self._thread_mgr.detector,
                        on_done=self._refresh_person_list)

    def _open_batch_import(self):
        BatchImportDialog(self.root, self._config, self._recognizer,
                          self._db, self._thread_mgr.detector,
                          on_done=self._refresh_person_list)

    def _toggle_alert(self, event=None):
        person = self._get_selected_person()
        if not person:
            messagebox.showinfo("提示", "請先選取一個人物")
            return
        name = person["name"]
        with self._alert_lock:
            if name in self._alert_names:
                self._alert_names.discard(name)
                status = "已關閉"
            else:
                self._alert_names.add(name)
                status = "已啟用"
        self._refresh_person_list()
        self._lbl_status.config(text=f"{name} 警報{status}")

    def _on_rename(self, event=None):
        person = self._get_selected_person()
        if not person:
            return
        old_name = person["name"]
        new_name = simpledialog.askstring(
            "重新命名", f"輸入新名稱（原名：{old_name}）：",
            initialvalue=old_name, parent=self.root,
        )
        if not new_name or new_name.strip() == old_name:
            return
        new_name = new_name.strip()
        if self._db.person_exists(new_name):
            messagebox.showerror("錯誤", f"名稱「{new_name}」已存在")
            return
        success = self._db.rename_person(person["id"], new_name)
        if success:
            with self._alert_lock:
                if old_name in self._alert_names:
                    self._alert_names.discard(old_name)
                    self._alert_names.add(new_name)
            self._recognizer.invalidate_name_cache()
            self._refresh_person_list()
        else:
            messagebox.showerror("錯誤", "重新命名失敗")

    def _on_rename_btn(self):
        self._on_rename()

    def _on_delete(self, event=None):
        person = self._get_selected_person()
        if not person:
            messagebox.showinfo("提示", "請先選取一個人物")
            return
        name = person["name"]
        if not messagebox.askyesno("確認刪除",
                                    f"確定要刪除「{name}」嗎？\n此操作無法還原。",
                                    parent=self.root):
            return
        try:
            with self._alert_lock:
                self._alert_names.discard(name)
            self._recognizer.remove_person(person["id"])
            # 重置緩衝區，設定冷卻期避免立即重新註冊
            self._unknown_buffer = self._empty_buffer()
            self._accumulate_timers.pop(person["id"], None)
            self._delete_cooldown_until = time.time() + 3.0
            # 刪除截圖資料夾
            try:
                screenshot_dir = self._config.SCREENSHOT_DIR / name
                if screenshot_dir.exists():
                    import shutil
                    shutil.rmtree(screenshot_dir)
            except Exception as e:
                logger.warning(f"[GUI] 刪除截圖資料夾失敗: {e}")
            self._refresh_person_list()
            self._lbl_status.config(text=f"已刪除: {name}")
        except Exception as e:
            logger.error(f"[GUI] 刪除人物失敗: {e}")
            messagebox.showerror("錯誤", f"刪除失敗: {e}")

    def _on_delete_all(self):
        persons = self._db.get_all_persons()
        if not persons:
            messagebox.showinfo("提示", "目前沒有任何人物")
            return
        if not messagebox.askyesno("確認全部刪除",
                                    f"確定要刪除全部 {len(persons)} 位人物嗎？\n此操作無法還原。",
                                    parent=self.root):
            return
        try:
            with self._alert_lock:
                self._alert_names.clear()
            self._recognizer.remove_all_persons()
            # 清理所有人物的截圖資料夾
            try:
                import shutil
                screenshot_base = self._config.SCREENSHOT_DIR
                if screenshot_base.exists():
                    for sub in screenshot_base.iterdir():
                        if sub.is_dir():
                            shutil.rmtree(sub)
            except Exception as e:
                logger.warning(f"[GUI] 清理截圖資料夾失敗: {e}")
            # 重置緩衝區和計時器，並設定冷卻期避免立即重新註冊
            self._unknown_buffer = self._empty_buffer()
            self._accumulate_timers.clear()
            self._delete_cooldown_until = time.time() + 3.0
            self._refresh_person_list()
            self._lbl_status.config(text=f"已刪除全部 {len(persons)} 位人物")
        except Exception as e:
            logger.error(f"[GUI] 全部刪除失敗: {e}")
            messagebox.showerror("錯誤", f"全部刪除失敗: {e}")

    def _on_camera_change(self):
        new_index = self._camera_var.get()
        if new_index == self._config.CAMERA_INDEX:
            return
        self._config.CAMERA_INDEX = new_index
        if self._is_running:
            self._async_restart_recognition()
        else:
            self._lbl_status.config(text=f"攝影機已切換至 {new_index}")

    def _async_restart_recognition(self):
        """在背景執行緒停止管線，避免阻塞 GUI。"""
        if self._restarting:
            return  # 防止重複觸發
        self._restarting = True
        self._is_running = False
        self._btn_toggle.config(text="切換中...", state=tk.DISABLED)
        self._lbl_status.config(text="正在切換攝影機...")

        def _do_restart():
            self._thread_mgr.stop_all(timeout=3.0)
            # 等待 OS 完全釋放攝影機資源
            time.sleep(0.5)
            self.root.after(0, self._finish_restart)

        threading.Thread(target=_do_restart, daemon=True).start()

    def _finish_restart(self):
        self._restarting = False
        self._btn_toggle.config(state=tk.NORMAL)
        self._start_recognition()

    def _show_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="重新命名", command=self._on_rename)
        menu.add_command(label="切換警示", command=self._toggle_alert)
        menu.add_separator()
        menu.add_command(label="刪除人物", command=self._on_delete)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _open_settings(self):
        # NOTE: 保存當前模型名，用於在回調中檢測是否改變
        self._prev_model_name = self._config.YOLO_MODEL_NAME
        SettingsDialog(self.root, self._config, on_apply=self._on_settings_applied)

    def _on_settings_applied(self):
        self._camera_var.set(self._config.CAMERA_INDEX)

        # NOTE: 檢查模型是否改變 — 必須和 _open_settings 保存的舊名比較
        # 因為 config 是同一對象，SettingsDialog 已經更新過了
        old_model = getattr(self, '_prev_model_name', None)
        if old_model and old_model != self._config.YOLO_MODEL_NAME:
            detector = self._thread_mgr.detector
            if detector:
                logger.info(f"[GUI] 模型已切換: {old_model} → {self._config.YOLO_MODEL_NAME}，正在重新載入...")
                detector.reload()

        if self._is_running:
            self._async_restart_recognition()

    def _on_threshold_change(self, val):
        v = round(float(val), 2)
        self._config.RECOGNITION_THRESHOLD = v
        self._lbl_threshold.config(text=f"{v:.2f}")

    def _on_close(self):
        logger.info("[GUI] 關閉視窗...")
        if self._is_running:
            self._is_running = False
            # 在背景停止執行緒，然後關閉視窗
            def _do_close():
                self._thread_mgr.stop_all(timeout=2.0)
                self.root.after(0, self.root.destroy)
            threading.Thread(target=_do_close, daemon=True).start()
        else:
            self.root.destroy()


# ──────────────────────────────────────────────
# 新增人物對話框
# ──────────────────────────────────────────────
class AddPersonDialog:
    def __init__(self, parent, config, recognizer, db, detector,
                 on_done: Optional[Callable] = None):
        self._config     = config
        self._recognizer = recognizer
        self._db         = db
        self._detector   = detector
        self._on_done    = on_done
        self._image_paths: list[str] = []

        self._win = tk.Toplevel(parent)
        self._win.title("新增人物")
        self._win.grab_set()
        self._win.resizable(False, False)
        self._build()

    def _build(self):
        pad = {"padx": 12, "pady": 6}

        tk.Label(self._win, text="姓名：").grid(row=0, column=0, sticky=tk.W, **pad)
        self._name_var = tk.StringVar()
        tk.Entry(self._win, textvariable=self._name_var,
                  width=24).grid(row=0, column=1, sticky=tk.W, **pad)

        tk.Label(self._win, text="部門（可選）：").grid(row=1, column=0, sticky=tk.W, **pad)
        self._dept_var = tk.StringVar()
        tk.Entry(self._win, textvariable=self._dept_var,
                  width=24).grid(row=1, column=1, sticky=tk.W, **pad)

        ttk.Button(self._win, text="選取照片",
                   command=self._select_images).grid(row=2, column=0, columnspan=2, **pad)

        self._lbl_photos = tk.Label(self._win, text="尚未選取任何照片")
        self._lbl_photos.grid(row=3, column=0, columnspan=2, **pad)

        self._progress = ttk.Progressbar(self._win, mode="determinate", length=260)
        self._progress.grid(row=4, column=0, columnspan=2, **pad)

        btn_frame = tk.Frame(self._win)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="確認新增",
                   command=self._confirm).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="取消",
                   command=self._win.destroy).pack(side=tk.LEFT, padx=4)

    def _select_images(self):
        paths = filedialog.askopenfilenames(
            parent=self._win,
            title="選取照片（可多選）",
            filetypes=[("圖片", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if paths:
            self._image_paths = list(paths)
            self._lbl_photos.config(text=f"已選取 {len(paths)} 張照片")

    def _confirm(self):
        name = self._name_var.get().strip()
        if not name:
            messagebox.showerror("錯誤", "請輸入姓名", parent=self._win)
            return
        if self._db.person_exists(name):
            messagebox.showerror("錯誤", f"名稱「{name}」已存在", parent=self._win)
            return
        if not self._image_paths:
            messagebox.showerror("錯誤", "請至少選取一張照片", parent=self._win)
            return
        dept = self._dept_var.get().strip() or None
        threading.Thread(target=self._process, args=(name, dept), daemon=True).start()

    def _process(self, name: str, dept: Optional[str]):
        try:
            person_id = self._db.add_person(name, department=dept)
            embeddings, qualities, sources = [], [], []
            total = len(self._image_paths)

            for i, path in enumerate(self._image_paths):
                self._win.after(0, lambda v=(i / total * 100): self._progress.config(value=v))
                frame = cv2.imread(path)
                if frame is None:
                    continue
                dets = self._detector.detect(frame)
                if not dets:
                    continue
                det = max(dets, key=lambda d: d.area)
                face_112 = self._recognizer.align_face(frame, det)
                emb = self._recognizer.extract_embedding(face_112)
                if emb is None:
                    continue
                quality = laplacian_variance(face_112)
                if quality < self._config.BATCH_LAPLACIAN_THRESHOLD:
                    continue
                embeddings.append(emb)
                qualities.append(quality)
                sources.append(path)
                try:
                    import shutil
                    dest_dir = self._config.SCREENSHOT_DIR / name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, dest_dir)
                except Exception:
                    pass

            if embeddings:
                self._recognizer.batch_add_faces(person_id, embeddings, sources, qualities)
                first = cv2.imread(sources[0])
                if first is not None:
                    self._db.set_thumbnail(person_id, encode_jpeg(first))
                msg = f"已新增「{name}」，共 {len(embeddings)} 個特徵向量"
            else:
                self._db.delete_person(person_id)
                msg = f"「{name}」所有照片均無法偵測到人臉，已取消新增"

            self._win.after(0, lambda: self._finish(msg, bool(embeddings)))
        except Exception as e:
            logger.error(f"[AddPerson] 失敗: {e}")
            self._win.after(0, lambda: messagebox.showerror("錯誤", str(e), parent=self._win))

    def _finish(self, message: str, success: bool):
        self._progress.config(value=100)
        messagebox.showinfo("完成", message, parent=self._win)
        if success and self._on_done:
            self._on_done()
        self._win.destroy()


# ──────────────────────────────────────────────
# 批量導入對話框
# ──────────────────────────────────────────────
class BatchImportDialog:
    def __init__(self, parent, config, recognizer, db, detector,
                 on_done: Optional[Callable] = None):
        self._config     = config
        self._recognizer = recognizer
        self._db         = db
        self._detector   = detector
        self._on_done    = on_done

        self._win = tk.Toplevel(parent)
        self._win.title("批量導入人臉")
        self._win.grab_set()
        self._win.resizable(False, False)
        self._build()

    def _build(self):
        pad = {"padx": 12, "pady": 8}
        info = ("選擇資料夾結構：\n"
                "  folder/\n"
                "    姓名A/  img1.jpg  img2.jpg ...\n"
                "    姓名B/  img1.jpg ...\n"
                "若資料夾內只有圖片（無子資料夾），\n"
                "則自動命名為「訪客N」。")
        tk.Label(self._win, text=info, justify=tk.LEFT).pack(**pad)

        ttk.Button(self._win, text="選擇資料夾",
                   command=self._select_folder).pack(**pad)

        self._lbl_folder = tk.Label(self._win, text="尚未選擇")
        self._lbl_folder.pack(**pad)

        self._progress = ttk.Progressbar(self._win, mode="determinate", length=320)
        self._progress.pack(**pad)

        self._log_text = tk.Text(self._win, height=8, width=46,
                                  font=("Consolas", 8), state=tk.DISABLED)
        self._log_text.pack(**pad)

        btn_frame = tk.Frame(self._win)
        btn_frame.pack(pady=8)
        self._btn_start = ttk.Button(btn_frame, text="開始導入",
                                      command=self._start, state=tk.DISABLED)
        self._btn_start.pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="關閉",
                   command=self._win.destroy).pack(side=tk.LEFT, padx=4)

        self._folder: Optional[Path] = None

    def _select_folder(self):
        folder = filedialog.askdirectory(parent=self._win, title="選擇人臉圖片資料夾")
        if folder:
            self._folder = Path(folder)
            self._lbl_folder.config(text=str(self._folder))
            self._btn_start.config(state=tk.NORMAL)

    def _log(self, msg: str):
        self._log_text.config(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.config(state=tk.DISABLED)

    def _start(self):
        if not self._folder:
            return
        self._btn_start.config(state=tk.DISABLED)
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        folder = self._folder
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def get_images(d: Path):
            return [f for f in d.iterdir()
                    if f.is_file() and f.suffix.lower() in img_extensions]

        if subdirs:
            tasks = [(d.name, get_images(d)) for d in subdirs]
        else:
            imgs = get_images(folder)
            if imgs:
                name = self._db.get_next_visitor_name()
                tasks = [(name, imgs)]
            else:
                self._win.after(0, lambda: self._log("找不到任何圖片"))
                return

        total_persons = len(tasks)
        success_count = 0
        skip_count = 0

        for pi, (person_name, img_paths) in enumerate(tasks):
            self._win.after(0, lambda n=person_name: self._log(f"\n處理：{n}"))
            progress_base = pi / total_persons * 100

            existing = self._db.get_person_by_name(person_name)
            if existing:
                person_id = existing["id"]
                self._win.after(0, lambda: self._log("  -> 已存在，追加特徵"))
            else:
                person_id = self._db.add_person(person_name)

            embeddings, qualities, sources = [], [], []

            for ii, img_path in enumerate(img_paths):
                sub_progress = progress_base + (ii / max(len(img_paths), 1)) * (100 / total_persons)
                self._win.after(0, lambda v=sub_progress: self._progress.config(value=v))

                frame = cv2.imread(str(img_path))
                if frame is None:
                    self._win.after(0, lambda p=img_path.name: self._log(f"  x 無法讀取: {p}"))
                    skip_count += 1
                    continue

                dets = self._detector.detect(frame)
                if not dets:
                    self._win.after(0, lambda p=img_path.name: self._log(f"  x 無人臉: {p}"))
                    skip_count += 1
                    continue

                det = max(dets, key=lambda d: d.area)
                face_112 = self._recognizer.align_face(frame, det)
                emb = self._recognizer.extract_embedding(face_112)
                if emb is None:
                    skip_count += 1
                    continue

                quality = laplacian_variance(face_112)
                if quality < self._config.BATCH_LAPLACIAN_THRESHOLD:
                    self._win.after(0, lambda p=img_path.name: self._log(f"  x 模糊: {p}"))
                    skip_count += 1
                    continue

                embeddings.append(emb)
                qualities.append(quality)
                sources.append(str(img_path))

                try:
                    import shutil
                    dest_dir = self._config.SCREENSHOT_DIR / person_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(img_path), dest_dir)
                except Exception:
                    pass

            if embeddings:
                self._recognizer.batch_add_faces(person_id, embeddings, sources, qualities)
                first = cv2.imread(sources[0])
                if first is not None:
                    self._db.set_thumbnail(person_id, encode_jpeg(first))
                self._win.after(0, lambda n=person_name, c=len(embeddings):
                                self._log(f"  OK {n}: {c} 個特徵向量"))
                success_count += 1
            else:
                self._db.delete_person(person_id)
                self._win.after(0, lambda n=person_name:
                                self._log(f"  x {n}: 無有效特徵"))

        summary = f"\n完成！成功 {success_count} 人，跳過 {skip_count} 張"
        self._win.after(0, lambda: self._finish(summary))

    def _finish(self, summary: str):
        self._progress.config(value=100)
        self._log(summary)
        if self._on_done:
            self._on_done()
        self._btn_start.config(state=tk.NORMAL)


# ──────────────────────────────────────────────
# 設定對話框（下拉選單版）
# ──────────────────────────────────────────────
class SettingsDialog:
    def __init__(self, parent, config, on_apply: Optional[Callable] = None):
        self._config = config
        self._on_apply = on_apply
        self._win = tk.Toplevel(parent)
        self._win.title("系統設定")
        self._win.grab_set()
        self._win.resizable(False, False)
        self._build()

    def _build(self):
        from config import YOLO_MODEL_OPTIONS, MODEL_DIR
        C = self._config
        pad = {"padx": 12, "pady": 5}
        row = 0

        # YOLO 模型大小（新增）
        tk.Label(self._win, text="YOLO模型：").grid(row=row, column=0, sticky=tk.W, **pad)
        model_names = list(YOLO_MODEL_OPTIONS.keys())
        # 找到当前选中的模型名
        current_model = "Nano (最快)"
        for name, filename in YOLO_MODEL_OPTIONS.items():
            if filename == C.YOLO_MODEL_NAME:
                current_model = name
                break
        self._model_var = tk.StringVar(value=current_model)
        model_cb = ttk.Combobox(self._win, textvariable=self._model_var,
                                 values=model_names, width=16, state="readonly")
        model_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # 攝影機索引（下拉選單）
        tk.Label(self._win, text="攝影機：").grid(row=row, column=0, sticky=tk.W, **pad)
        self._camera_var = tk.StringVar(value=str(C.CAMERA_INDEX))
        cam_cb = ttk.Combobox(self._win, textvariable=self._camera_var,
                               values=[str(i) for i in range(10)],
                               width=6, state="readonly")
        cam_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # 辨識閾值（下拉選單）
        tk.Label(self._win, text="辨識閾值：").grid(row=row, column=0, sticky=tk.W, **pad)
        threshold_values = [f"{v/100:.2f}" for v in range(20, 86, 5)]
        self._threshold_var = tk.StringVar(value=f"{C.RECOGNITION_THRESHOLD:.2f}")
        thr_cb = ttk.Combobox(self._win, textvariable=self._threshold_var,
                               values=threshold_values, width=6, state="readonly")
        thr_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # 跳幀數量（下拉選單）
        tk.Label(self._win, text="跳幀數量：").grid(row=row, column=0, sticky=tk.W, **pad)
        self._skip_var = tk.StringVar(value=str(C.SKIP_FRAMES))
        skip_cb = ttk.Combobox(self._win, textvariable=self._skip_var,
                                values=["1", "2", "3", "4", "5"],
                                width=6, state="readonly")
        skip_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # 檢測影像尺寸（下拉選單）
        tk.Label(self._win, text="檢測解析度：").grid(row=row, column=0, sticky=tk.W, **pad)
        self._imgsz_var = tk.StringVar(value=str(C.DETECT_IMG_SIZE))
        imgsz_cb = ttk.Combobox(self._win, textvariable=self._imgsz_var,
                                 values=["320", "480", "640", "800", "1024"],
                                 width=6, state="readonly")
        imgsz_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # 推理裝置（下拉選單）
        tk.Label(self._win, text="推理裝置：").grid(row=row, column=0, sticky=tk.W, **pad)
        self._device_var = tk.StringVar(value=C.DETECT_DEVICE)
        dev_cb = ttk.Combobox(self._win, textvariable=self._device_var,
                               values=["cpu", "cuda:0", "mps"],
                               width=8, state="readonly")
        dev_cb.grid(row=row, column=1, sticky=tk.W, **pad)
        row += 1

        # NOTE: 模型切换性能警告
        self._lbl_warning = tk.Label(
            self._win, text="※ 較大模型需要更多資源，CPU 上可能降低畫面速率",
            fg="gray", font=("Helvetica", 9))
        self._lbl_warning.grid(row=row, column=0, columnspan=2, **pad)
        row += 1

        # 按鈕
        btn_frame = tk.Frame(self._win)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="套用",
                   command=self._apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="取消",
                   command=self._win.destroy).pack(side=tk.LEFT, padx=4)

    def _apply(self):
        from config import YOLO_MODEL_OPTIONS, MODEL_DIR
        C = self._config
        try:
            C.CAMERA_INDEX          = int(self._camera_var.get())
            C.RECOGNITION_THRESHOLD = float(self._threshold_var.get())
            C.SKIP_FRAMES           = int(self._skip_var.get())
            C.DETECT_IMG_SIZE       = int(self._imgsz_var.get())
            C.DETECT_DEVICE         = self._device_var.get()

            # NOTE: 更新 YOLO 模型名和路径
            model_label = self._model_var.get()
            if model_label in YOLO_MODEL_OPTIONS:
                new_model_name = YOLO_MODEL_OPTIONS[model_label]
                if new_model_name != C.YOLO_MODEL_NAME:
                    C.YOLO_MODEL_NAME = new_model_name
                    C.YOLO_MODEL_PATH = MODEL_DIR / new_model_name
                    logger.info(f"[Settings] 模型切換至: {model_label} ({new_model_name})")

            self._win.destroy()
            if self._on_apply:
                self._on_apply()
        except ValueError as e:
            messagebox.showerror("錯誤", f"輸入格式有誤: {e}", parent=self._win)

