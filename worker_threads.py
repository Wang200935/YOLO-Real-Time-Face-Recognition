"""
worker_threads.py — 多執行緒 Pipeline
架構：CameraThread → DetectionThread → RecognitionThread → GUI 主執行緒
                                      ↘ DBWriterThread

每個執行緒透過 queue.Queue 傳遞資料，shutdown_event 統一控制停止。
"""

import logging
import platform
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 資料結構
# ──────────────────────────────────────────────
@dataclass
class RecognitionResult:
    """單一人臉辨識結果，從辨識執行緒傳遞至 GUI。"""
    # 人體 bounding box
    x1: int
    y1: int
    x2: int
    y2: int
    name: str
    confidence: float
    person_id: Optional[int]
    is_unknown: bool
    is_alert: bool = False
    embedding: Optional[np.ndarray] = None   # 用於自動新增未知人物
    face_aligned: Optional[np.ndarray] = None # 對齊後 112x112 BGR，用於品質檢查與截圖
    quality: float = 0.0                      # Laplacian 清晰度分數
    # 人體骨架 (17,3)=[x,y,conf]，用於 GUI 繪製骨架
    body_keypoints: Optional[np.ndarray] = None
    # 從骨架衍生的人臉區域
    face_x1: int = 0
    face_y1: int = 0
    face_x2: int = 0
    face_y2: int = 0


@dataclass
class FrameResult:
    """一幀的所有辨識結果，放入 result_queue。"""
    frame: np.ndarray
    results: list[RecognitionResult]
    timestamp: float = field(default_factory=time.time)


@dataclass
class DBWriteTask:
    """非同步資料庫寫入任務。"""
    task_type: str   # "log_recognition" | "save_screenshot"
    kwargs: dict     = field(default_factory=dict)


# ──────────────────────────────────────────────
# 攝影機執行緒
# ──────────────────────────────────────────────
class CameraThread(threading.Thread):
    """
    持續讀取攝影機幀，放入 frame_queue。
    使用 put_nowait() 丟棄積壓的舊幀（避免延遲累積）。
    支援外部 force_release() 強制釋放攝影機。
    """

    def __init__(self, frame_queue: queue.Queue,
                 shutdown_event: threading.Event, config):
        super().__init__(name="CameraThread", daemon=True)
        self._frame_queue = frame_queue
        self._shutdown = shutdown_event
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self.error: Optional[str] = None

    def run(self):
        backend = self._get_backend()
        cam_idx = self._config.CAMERA_INDEX
        logger.info(f"[Camera] 開啟攝影機 index={cam_idx}, backend={backend}")

        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened():
            self.error = f"無法開啟攝影機 (index={cam_idx})"
            logger.error(f"[Camera] {self.error}")
            self._frame_queue.put(None)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          self._config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        with self._cap_lock:
            self._cap = cap
        logger.info("[Camera] 攝影機啟動成功")

        try:
            while not self._shutdown.is_set():
                with self._cap_lock:
                    if self._cap is None:
                        break
                    grabbed = self._cap.grab()
                if not grabbed:
                    if self._shutdown.is_set():
                        break
                    time.sleep(0.05)
                    continue
                with self._cap_lock:
                    if self._cap is None:
                        break
                    ret, frame = self._cap.retrieve()
                if not ret:
                    continue
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
        finally:
            self._release_cap()
            logger.info("[Camera] 攝影機已釋放")
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                pass

    def _release_cap(self):
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def force_release(self):
        """從外部執行緒強制釋放攝影機，讓 run() 迴圈立即退出。"""
        logger.info("[Camera] 外部強制釋放攝影機")
        self._release_cap()

    def _get_backend(self) -> int:
        system = platform.system()
        if system == "Windows":
            return cv2.CAP_DSHOW
        elif system == "Darwin":
            return cv2.CAP_AVFOUNDATION
        else:
            return cv2.CAP_V4L2 if platform.machine().lower() in ("armv7l", "aarch64") \
                   else cv2.CAP_ANY


# ──────────────────────────────────────────────
# 檢測執行緒
# ──────────────────────────────────────────────
class DetectionThread(threading.Thread):
    """
    從 frame_queue 取幀，執行 YOLOv11 人臉檢測，
    將結果放入 detection_queue。
    """

    def __init__(self, frame_queue: queue.Queue,
                 detection_queue: queue.Queue,
                 detector,
                 shutdown_event: threading.Event):
        super().__init__(name="DetectionThread", daemon=False)
        self._frame_q = frame_queue
        self._detect_q = detection_queue
        self._detector = detector
        self._shutdown = shutdown_event
        self.detect_timer_ms: float = 0.0

    def run(self):
        logger.info("[Detect] 檢測執行緒啟動")
        first_frame = True
        while not self._shutdown.is_set():
            try:
                frame = self._frame_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:  # sentinel
                self._detect_q.put(None)
                logger.info("[Detect] 收到 sentinel，結束")
                break

            t0 = time.perf_counter()
            try:
                detections = self._detector.detect(frame)
            except Exception as e:
                logger.error(f"[Detect] 檢測異常: {e}")
                detections = []
            self.detect_timer_ms = (time.perf_counter() - t0) * 1000

            if first_frame:
                logger.info(f"[Detect] 第一幀處理完成: {len(detections)} 張臉, {self.detect_timer_ms:.0f}ms")
                first_frame = False

            try:
                self._detect_q.put_nowait({
                    "frame": frame,
                    "detections": detections,
                    "timestamp": time.time(),
                })
            except queue.Full:
                pass  # 丟棄以避免積壓

        logger.info("[Detect] 檢測執行緒結束")


# ──────────────────────────────────────────────
# 辨識執行緒
# ──────────────────────────────────────────────
class RecognitionThread(threading.Thread):
    """
    從 detection_queue 取檢測結果，
    執行人臉對齊、特徵提取、餘弦相似度辨識，
    將 FrameResult 放入 result_queue，
    並將辨識記錄推入 db_write_queue。
    """

    def __init__(self, detection_queue: queue.Queue,
                 result_queue: queue.Queue,
                 db_write_queue: queue.Queue,
                 recognizer,
                 alert_names: set,        # 警報人物集合（共享，需外部加鎖）
                 alert_lock: threading.Lock,
                 shutdown_event: threading.Event):
        super().__init__(name="RecognitionThread", daemon=False)
        self._detect_q    = detection_queue
        self._result_q    = result_queue
        self._db_write_q  = db_write_queue
        self._recognizer  = recognizer
        self._alert_names = alert_names
        self._alert_lock  = alert_lock
        self._shutdown    = shutdown_event
        self.recog_timer_ms: float = 0.0

    def run(self):
        logger.info("[Recognize] 辨識執行緒啟動")
        first_frame = True
        while not self._shutdown.is_set():
            try:
                item = self._detect_q.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # sentinel
                self._result_q.put(None)
                logger.info("[Recognize] 收到 sentinel，結束")
                break

            if first_frame:
                logger.info("[Recognize] 收到第一幀，開始辨識處理")
                first_frame = False

            frame       = item["frame"]
            detections  = item["detections"]
            timestamp   = item["timestamp"]

            t0 = time.perf_counter()
            results = []

            for det in detections:
                try:
                    # 必須有人臉區域才進行辨識（避免用全身裁切產生垃圾 embedding）
                    if not det.has_face:
                        # 沒有偵測到臉部關鍵點，只保留骨架顯示，不做辨識
                        results.append(RecognitionResult(
                            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                            name="--", confidence=det.confidence,
                            person_id=None, is_unknown=True,
                            body_keypoints=det.body_keypoints,
                            face_x1=det.face_x1, face_y1=det.face_y1,
                            face_x2=det.face_x2, face_y2=det.face_y2,
                        ))
                        continue

                    face_112 = self._recognizer.align_face(frame, det)
                    embedding = self._recognizer.extract_embedding(face_112)
                    if embedding is None:
                        continue

                    # 計算清晰度（Laplacian 方差）
                    gray = cv2.cvtColor(face_112, cv2.COLOR_BGR2GRAY)
                    quality = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                    # 品質太低（模糊）的臉不辨識，避免錯誤匹配
                    if quality < 30.0:
                        results.append(RecognitionResult(
                            x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                            name="--", confidence=det.confidence,
                            person_id=None, is_unknown=True,
                            body_keypoints=det.body_keypoints,
                            face_x1=det.face_x1, face_y1=det.face_y1,
                            face_x2=det.face_x2, face_y2=det.face_y2,
                            quality=quality,
                        ))
                        continue

                    person_id, name, confidence = self._recognizer.identify_with_id(embedding)
                    is_unknown = (person_id is None)

                    # 判斷是否為警報人物
                    is_alert = False
                    if not is_unknown:
                        with self._alert_lock:
                            is_alert = name in self._alert_names

                    res = RecognitionResult(
                        x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                        name=name,
                        confidence=confidence,
                        person_id=person_id,
                        is_unknown=is_unknown,
                        is_alert=is_alert,
                        embedding=embedding,
                        face_aligned=face_112,
                        quality=quality,
                        body_keypoints=det.body_keypoints,
                        face_x1=det.face_x1, face_y1=det.face_y1,
                        face_x2=det.face_x2, face_y2=det.face_y2,
                    )
                    results.append(res)

                    # 非同步記錄至資料庫
                    try:
                        self._db_write_q.put_nowait(DBWriteTask(
                            task_type="log_recognition",
                            kwargs={
                                "person_id": person_id,
                                "confidence": confidence,
                                "is_unknown": is_unknown,
                            }
                        ))
                    except queue.Full:
                        pass

                except Exception as e:
                    logger.error(f"[Recognize] 單一人臉處理異常: {e}")

            self.recog_timer_ms = (time.perf_counter() - t0) * 1000

            frame_result = FrameResult(
                frame=frame, results=results, timestamp=timestamp
            )
            try:
                self._result_q.put_nowait(frame_result)
            except queue.Full:
                pass

        logger.info("[Recognize] 辨識執行緒結束")


# ──────────────────────────────────────────────
# 資料庫寫入執行緒
# ──────────────────────────────────────────────
class DBWriterThread(threading.Thread):
    """
    非同步處理資料庫寫入任務，避免阻塞辨識執行緒。
    擁有自己獨立的 DB 連線（threading.local 機制）。
    """

    def __init__(self, db_write_queue: queue.Queue,
                 db_manager,
                 shutdown_event: threading.Event):
        super().__init__(name="DBWriterThread", daemon=False)
        self._queue   = db_write_queue
        self._db      = db_manager
        self._shutdown = shutdown_event

    def run(self):
        logger.info("[DBWriter] 資料庫寫入執行緒啟動")
        while not self._shutdown.is_set():
            try:
                task: DBWriteTask = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:  # sentinel
                logger.info("[DBWriter] 收到 sentinel，結束")
                break

            try:
                self._handle_task(task)
            except Exception as e:
                logger.error(f"[DBWriter] 任務執行失敗: {e}")

        # 排空佇列剩餘任務
        while not self._queue.empty():
            try:
                task = self._queue.get_nowait()
                if task:
                    self._handle_task(task)
            except (queue.Empty, Exception):
                break

        logger.info("[DBWriter] 資料庫寫入執行緒結束")

    def _handle_task(self, task: DBWriteTask):
        if task.task_type == "log_recognition":
            self._db.log_recognition(**task.kwargs)
        elif task.task_type == "save_screenshot":
            # 未來可擴充截圖儲存邏輯
            pass
        else:
            logger.warning(f"[DBWriter] 未知任務類型: {task.task_type}")


# ──────────────────────────────────────────────
# 執行緒管理器
# ──────────────────────────────────────────────
class ThreadManager:
    """
    統一管理所有工作執行緒的生命週期。
    提供 start_all() 和 stop_all() 介面。
    """

    def __init__(self, config, detector, recognizer, db_manager,
                 result_queue: queue.Queue,
                 alert_names: set, alert_lock: threading.Lock):
        self._config     = config
        self._detector   = detector
        self._recognizer = recognizer
        self._db         = db_manager
        self._result_q   = result_queue
        self._alert_names = alert_names
        self._alert_lock  = alert_lock

        self._shutdown = threading.Event()
        self._frame_q  = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
        self._detect_q = queue.Queue(maxsize=config.DETECT_QUEUE_SIZE)
        self._db_write_q = queue.Queue(maxsize=config.DB_WRITE_QUEUE_SIZE)

        self._threads: list[threading.Thread] = []
        self.camera_thread: Optional[CameraThread] = None
        self.detect_thread: Optional[DetectionThread] = None
        self.recog_thread: Optional[RecognitionThread] = None
        self.db_thread: Optional[DBWriterThread] = None

    def start_all(self):
        """建立並啟動所有工作執行緒。"""
        # 確保舊執行緒全部結束（防止 shutdown Event 衝突）
        for t in self._threads:
            if t.is_alive():
                logger.warning(f"[ThreadManager] 等待舊執行緒 {t.name} 結束...")
                t.join(timeout=2.0)
                if t.is_alive():
                    logger.error(f"[ThreadManager] 舊執行緒 {t.name} 仍在運行，強制繼續")
        self._threads.clear()

        self._shutdown.clear()

        # 清空殘留的 sentinel / 舊資料
        for q in [self._frame_q, self._detect_q, self._result_q, self._db_write_q]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self._detector.reset_frame_count()

        self.camera_thread = CameraThread(
            self._frame_q, self._shutdown, self._config)
        self.detect_thread = DetectionThread(
            self._frame_q, self._detect_q, self._detector, self._shutdown)
        self.recog_thread = RecognitionThread(
            self._detect_q, self._result_q, self._db_write_q,
            self._recognizer, self._alert_names, self._alert_lock,
            self._shutdown)
        self.db_thread = DBWriterThread(
            self._db_write_q, self._db, self._shutdown)

        self._threads = [
            self.camera_thread,
            self.detect_thread,
            self.recog_thread,
            self.db_thread,
        ]

        for t in self._threads:
            t.start()

        logger.info("[ThreadManager] 所有執行緒已啟動")

    def stop_all(self, timeout: float = 3.0):
        """
        優雅停止所有執行緒：
        1. 設定 shutdown_event
        2. 強制釋放攝影機（打斷 cap.grab 阻塞）
        3. 向各 queue 放入 sentinel None
        4. join 等待
        """
        logger.info("[ThreadManager] 停止所有執行緒…")
        self._shutdown.set()

        # 強制釋放攝影機，讓 CameraThread 立即退出
        if self.camera_thread is not None:
            self.camera_thread.force_release()

        # 送出 sentinel 解鎖可能在 queue.get() 阻塞的執行緒
        for q in [self._frame_q, self._detect_q, self._result_q, self._db_write_q]:
            # 先嘗試清一個位置再放 sentinel
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

        for t in self._threads:
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning(f"[ThreadManager] 執行緒 {t.name} 未在 {timeout}s 內結束")

        self._threads.clear()
        logger.info("[ThreadManager] 所有執行緒已結束")

    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    @property
    def detect_ms(self) -> float:
        return self.detect_thread.detect_timer_ms if self.detect_thread else 0.0

    @property
    def recog_ms(self) -> float:
        return self.recog_thread.recog_timer_ms if self.recog_thread else 0.0

    @property
    def detector(self):
        return self._detector

    @property
    def recognizer(self):
        return self._recognizer
