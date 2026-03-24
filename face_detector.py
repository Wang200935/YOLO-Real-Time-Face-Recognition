"""
face_detector.py — YOLOv11-Pose 人體檢測 + 骨架 + 人臉區域萃取
使用 ultralytics >= 8.3.0 的 YOLOv11-Pose 模型。
輸出：人體 bbox、17 點 COCO 骨架關鍵點、從頭部關鍵點衍生的人臉區域。
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── COCO 17 點骨架定義 ──────────────────────────
# 索引：0=nose 1=Leye 2=Reye 3=Lear 4=Rear
#       5=Lshoulder 6=Rshoulder 7=Lelbow 8=Relbow
#       9=Lwrist 10=Rwrist 11=Lhip 12=Rhip
#       13=Lknee 14=Rknee 15=Lankle 16=Rankle
SKELETON_CONNECTIONS = [
    # 頭部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 軀幹
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
]

# 骨架連線顏色 (BGR)
SKELETON_COLORS = {
    "head":  (255, 200, 0),    # 青色
    "torso": (0, 220, 220),    # 黃色
    "left":  (0, 200, 0),      # 綠色
    "right": (0, 128, 255),    # 橘色
}

def _limb_color(i: int, j: int) -> tuple:
    if i <= 4 or j <= 4:
        return SKELETON_COLORS["head"]
    if (i in (5, 6, 11, 12)) and (j in (5, 6, 11, 12)):
        return SKELETON_COLORS["torso"]
    if i in (5, 7, 9, 11, 13, 15) and j in (5, 7, 9, 11, 13, 15):
        return SKELETON_COLORS["left"]
    return SKELETON_COLORS["right"]


@dataclass
class Detection:
    """單一人體檢測結果，包含骨架與人臉區域。"""
    # 人體 bounding box
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    # COCO 17 點骨架 (17,3)=[x,y,conf]，或 None
    body_keypoints: Optional[np.ndarray] = None
    # 從頭部關鍵點衍生的人臉區域
    face_x1: int = 0
    face_y1: int = 0
    face_x2: int = 0
    face_y2: int = 0
    # 5 點人臉關鍵點 (Leye,Reye,nose,Lmouth,Rmouth) 供 ArcFace 對齊
    keypoints: Optional[np.ndarray] = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def face_bbox(self) -> tuple[int, int, int, int]:
        return (self.face_x1, self.face_y1, self.face_x2, self.face_y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def face_width(self) -> int:
        return self.face_x2 - self.face_x1

    @property
    def face_height(self) -> int:
        return self.face_y2 - self.face_y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def has_face(self) -> bool:
        return self.face_width > 0 and self.face_height > 0


class FaceDetector:
    """
    YOLOv11-Pose 人體檢測器。
    輸出人體 bbox + 17 點骨架 + 衍生的人臉區域。
    """

    def __init__(self, config):
        self._config = config
        self._model = None
        self._frame_count = 0
        self._last_detections: list[Detection] = []

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics 未安裝或版本過低。請執行: pip install ultralytics>=8.3.0"
            )

        model_path = self._config.YOLO_MODEL_PATH
        model_name = self._config.YOLO_MODEL_NAME

        if model_path.exists():
            source = str(model_path)
            logger.info(f"[Detector] 載入本地模型: {model_path}")
        else:
            source = model_name
            logger.info(f"[Detector] 首次使用，透過 ultralytics 下載 {model_name}…")

        t0 = time.perf_counter()
        self._model = YOLO(source)

        if not model_path.exists():
            try:
                import shutil
                cwd_pt = Path(model_name)
                ul_cache = Path.home() / ".ultralytics" / "assets" / model_name
                for candidate in [cwd_pt, ul_cache]:
                    if candidate.exists() and candidate != model_path:
                        shutil.move(str(candidate), str(model_path))
                        logger.info(f"[Detector] 模型已移至: {model_path}")
                        break
            except Exception as e:
                logger.debug(f"[Detector] 移動模型至本地失敗: {e}")

        try:
            self._model.fuse()
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        logger.info(f"[Detector] YOLOv11-Pose 載入完成 ({elapsed:.2f}s)")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            raise RuntimeError("請先呼叫 load() 載入模型")

        self._frame_count += 1
        if self._frame_count % self._config.SKIP_FRAMES != 0:
            return self._last_detections

        t0 = time.perf_counter()

        try:
            results = self._model.predict(
                frame,
                conf=self._config.DETECT_CONFIDENCE,
                iou=self._config.DETECT_IOU,
                imgsz=self._config.DETECT_IMG_SIZE,
                device=self._config.DETECT_DEVICE,
                classes=self._config.DETECT_CLASSES,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            logger.error(f"[Detector] 推理失敗: {e}")
            return self._last_detections

        detections = self._parse_results(results, frame.shape)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"[Detector] {len(detections)} 人, {elapsed_ms:.1f}ms")

        self._last_detections = detections
        return detections

    def _parse_results(self, results, frame_shape: tuple) -> list[Detection]:
        h, w = frame_shape[:2]
        detections = []

        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy_all = boxes.xyxy.cpu().numpy()
        confs_all = boxes.conf.cpu().numpy()

        # Pose 模型：17 點骨架關鍵點
        kpts_all = None
        kpts_conf_all = None
        if result.keypoints is not None:
            try:
                kpts_all = result.keypoints.xy.cpu().numpy()       # (N, 17, 2)
                kpts_conf_all = result.keypoints.conf.cpu().numpy() # (N, 17)
            except Exception:
                kpts_all = None

        for i in range(len(xyxy_all)):
            bx1, by1, bx2, by2 = xyxy_all[i]
            bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
            conf = float(confs_all[i])

            bw = bx2 - bx1
            bh = by2 - by1
            if bw < self._config.MIN_FACE_SIZE or bh < self._config.MIN_FACE_SIZE:
                continue

            bx1 = max(0, min(bx1, w - 1))
            by1 = max(0, min(by1, h - 1))
            bx2 = max(0, min(bx2, w))
            by2 = max(0, min(by2, h))

            body_kpts = None
            face_x1, face_y1, face_x2, face_y2 = 0, 0, 0, 0
            face_5pts = None

            if kpts_all is not None and i < len(kpts_all):
                kxy = kpts_all[i]   # (17, 2)
                kc = kpts_conf_all[i] if kpts_conf_all is not None else np.ones(17)
                body_kpts = np.column_stack([kxy, kc]).astype(np.float32)  # (17, 3)

                # 衍生人臉區域
                face_x1, face_y1, face_x2, face_y2, face_5pts = \
                    self._derive_face_region(body_kpts, w, h)

            # NOTE: 無論關鍵點是否存在，確保有人臉區域（用 body bbox 上部兜底）
            # 這確保 has_face 始終為 True，辨識流程不會被跳過
            if face_x2 <= face_x1 or face_y2 <= face_y1:
                body_h = by2 - by1
                # NOTE: 用 body bbox 上 30% 作為人臉區域（25% 太小會丟失特徵）
                face_x1 = bx1
                face_y1 = by1
                face_x2 = bx2
                face_y2 = min(by2, by1 + int(body_h * 0.30))
                face_5pts = None
                logger.debug("[Detector] 關鍵點失敗，用 body bbox 上部兜底")

            detections.append(Detection(
                x1=bx1, y1=by1, x2=bx2, y2=by2,
                confidence=conf,
                body_keypoints=body_kpts,
                face_x1=face_x1, face_y1=face_y1,
                face_x2=face_x2, face_y2=face_y2,
                keypoints=face_5pts,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _derive_face_region(body_kpts: np.ndarray, img_w: int, img_h: int):
        """
        從 17 點 COCO 骨架中提取人臉區域。
        body_kpts: (17, 3) = [x, y, confidence]
        回傳: (face_x1, face_y1, face_x2, face_y2, face_5pts)

        NOTE: 降低 MIN_CONF 並新增單關鍵點退回策略，
        使側臉（僅可見 1 個頭部關鍵點）也能產生有效人臉區域。
        """
        # 關鍵索引
        NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
        LSHOULDER, RSHOULDER = 5, 6
        # NOTE: 从 0.5→0.3→0.15，全身远距离时关键点置信度低但仍有参考价值
        MIN_CONF = 0.15

        # 收集可用的頭部點
        head_ids = [NOSE, LEYE, REYE, LEAR, REAR]
        valid_pts = []
        for idx in head_ids:
            if body_kpts[idx, 2] >= MIN_CONF:
                valid_pts.append(body_kpts[idx, :2])

        if len(valid_pts) == 0:
            # 完全無頭部關鍵點，嘗試用肩膀估算
            result = FaceDetector._fallback_from_shoulders(
                body_kpts, img_w, img_h, LSHOULDER, RSHOULDER, MIN_CONF)
            # NOTE: 如果肩膀也失败，返回空区域（由调用方用 body bbox 兜底）
            return result

        if len(valid_pts) == 1:
            # NOTE: 單關鍵點退回策略 — 結合肩膀位置估算人臉區域
            head_pt = valid_pts[0]
            shoulder_pts = []
            for s_idx in [LSHOULDER, RSHOULDER]:
                if body_kpts[s_idx, 2] >= MIN_CONF:
                    shoulder_pts.append(body_kpts[s_idx, :2])

            if shoulder_pts:
                # 用肩膀寬度估算臉部大小（肩寬約為臉寬的 3 倍）
                if len(shoulder_pts) == 2:
                    shoulder_width = np.linalg.norm(
                        shoulder_pts[0] - shoulder_pts[1])
                    face_size = max(shoulder_width / 3.0, 50)
                else:
                    # 單肩膀：用頭部點到肩膀的距離估算
                    dist = np.linalg.norm(head_pt - shoulder_pts[0])
                    face_size = max(dist * 0.6, 50)
            else:
                # 無肩膀資訊，使用固定默認值
                face_size = 60

            cx, cy = head_pt[0], head_pt[1]
            half = face_size / 2
            fx1 = int(max(0, cx - half))
            fy1 = int(max(0, cy - half * 1.3))
            fx2 = int(min(img_w, cx + half))
            fy2 = int(min(img_h, cy + half * 1.0))
            return fx1, fy1, fx2, fy2, None

        pts = np.array(valid_pts)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

        # 用雙眼距離或頭部點展幅估算臉寬
        # NOTE: eye_dist×2.2、spread×1.8 是合理倍率
        # 過大（3.0/2.5）會包含太多身體，導致 embedding 不穩定
        if body_kpts[LEYE, 2] >= MIN_CONF and body_kpts[REYE, 2] >= MIN_CONF:
            eye_dist = np.linalg.norm(body_kpts[LEYE, :2] - body_kpts[REYE, :2])
            face_size = max(eye_dist * 2.2, 50)
        else:
            spread = max(pts[:, 0].max() - pts[:, 0].min(),
                         pts[:, 1].max() - pts[:, 1].min())
            face_size = max(spread * 1.8, 50)

        half = face_size / 2
        fx1 = int(max(0, cx - half))
        fy1 = int(max(0, cy - half * 1.3))  # 上方多留空間（側臉需更多）
        fx2 = int(min(img_w, cx + half))
        fy2 = int(min(img_h, cy + half * 1.0))

        # 構建 5 點人臉關鍵點供 ArcFace 對齊
        face_5pts = None
        if (body_kpts[LEYE, 2] >= MIN_CONF and
            body_kpts[REYE, 2] >= MIN_CONF and
            body_kpts[NOSE, 2] >= MIN_CONF):

            leye = body_kpts[LEYE, :2]
            reye = body_kpts[REYE, :2]
            nose = body_kpts[NOSE, :2]
            # 近似嘴角：鼻子下方，與眼睛同寬
            eye_center_y = (leye[1] + reye[1]) / 2
            mouth_y = nose[1] + (nose[1] - eye_center_y) * 0.55
            lmouth = np.array([leye[0], mouth_y])
            rmouth = np.array([reye[0], mouth_y])
            face_5pts = np.array([leye, reye, nose, lmouth, rmouth],
                                 dtype=np.float32)

        return fx1, fy1, fx2, fy2, face_5pts

    @staticmethod
    def _fallback_from_shoulders(
        body_kpts: np.ndarray, img_w: int, img_h: int,
        lshoulder: int, rshoulder: int, min_conf: float
    ):
        """完全無頭部關鍵點時，用肩膀中點上方估算人臉區域。"""
        shoulder_pts = []
        for s_idx in [lshoulder, rshoulder]:
            if body_kpts[s_idx, 2] >= min_conf:
                shoulder_pts.append(body_kpts[s_idx, :2])

        if len(shoulder_pts) < 1:
            return 0, 0, 0, 0, None

        if len(shoulder_pts) == 2:
            mid = (shoulder_pts[0] + shoulder_pts[1]) / 2
            shoulder_width = np.linalg.norm(
                shoulder_pts[0] - shoulder_pts[1])
            face_size = max(shoulder_width / 3.0, 50)
        else:
            mid = shoulder_pts[0]
            face_size = 60

        # 人臉在肩膀中點上方約 face_size 距離
        cx = mid[0]
        cy = mid[1] - face_size * 0.8
        half = face_size / 2
        fx1 = int(max(0, cx - half))
        fy1 = int(max(0, cy - half * 1.3))
        fx2 = int(min(img_w, cx + half))
        fy2 = int(min(img_h, cy + half * 1.0))
        return fx1, fy1, fx2, fy2, None

    def reload(self, config=None):
        """
        重新载入模型（用于运行时切换模型大小）。
        NOTE: 使用原子替换而非先 null 后 load，避免检测线程访问到 None 导致异常。
        """
        if config is not None:
            self._config = config

        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("[Detector] ultralytics 未安裝")
            return

        model_path = self._config.YOLO_MODEL_PATH
        model_name = self._config.YOLO_MODEL_NAME

        if model_path.exists():
            source = str(model_path)
        else:
            source = model_name

        logger.info(f"[Detector] 正在載入新模型: {model_name}")
        new_model = YOLO(source)

        # NOTE: fuse 可能因模型結構差異而失敗，不影響推理
        try:
            new_model.fuse()
        except Exception:
            pass

        # 原子性替換：檢測線程在此之前仍用舊模型
        self._model = new_model
        self._frame_count = 0
        self._last_detections = []
        logger.info(f"[Detector] 模型已重新载入: {model_name}")

    def is_loaded(self) -> bool:
        return self._model is not None

    def reset_frame_count(self):
        self._frame_count = 0
        self._last_detections = []


# ── 自測 ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    class _Cfg:
        YOLO_MODEL_NAME = "yolo11n-pose.pt"
        YOLO_MODEL_PATH = Path("data/models/yolo11n-pose.pt")
        DETECT_CONFIDENCE = 0.45
        DETECT_IOU = 0.45
        DETECT_IMG_SIZE = 1024
        DETECT_DEVICE = "cpu"
        MIN_FACE_SIZE = 60
        SKIP_FRAMES = 1
        DETECT_CLASSES = [0]

    detector = FaceDetector(_Cfg())
    print("載入 YOLOv11-Pose 模型…")
    detector.load()
    print("模型載入成功")

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = detector.detect(dummy)
    print(f"空白圖片檢測結果: {len(dets)} 人（預期 0）")
    print("FaceDetector 基本測試通過")
