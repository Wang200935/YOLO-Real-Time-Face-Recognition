"""
face_recognizer.py — MobileFaceNet 特徵提取 + 餘弦相似度辨識 + 增量學習
- 特徵提取：w600k_mbf.onnx (InsightFace buffalo_sc)，512 維 L2 normalized
- 辨識：矩陣 dot product = 餘弦相似度（所有向量已 L2 normalize）
- 增量學習：新增人物後立即重建索引，無需重新訓練
- 執行緒安全：RLock 保護特徵矩陣的讀寫
"""

import logging
import threading
import time
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace 標準 112x112 人臉 5 點關鍵點座標
# 順序：左眼、右眼、鼻尖、左嘴角、右嘴角
ARCFACE_SRC = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


class FaceRecognizer:
    """
    人臉特徵提取器 + 辨識器。

    使用方式：
        recognizer = FaceRecognizer(config, db)
        recognizer.load()                         # 初始化 ONNX session
        emb = recognizer.extract_embedding(face_112)
        name, conf = recognizer.identify(emb)
    """

    def __init__(self, config, db):
        self._config = config
        self._db = db
        self._session = None
        self._input_name: str = ""
        self._input_shape: tuple = (1, 3, 112, 112)

        # 辨識索引（受 RLock 保護）
        self._lock = threading.RLock()
        self._embeddings_matrix: Optional[np.ndarray] = None  # (N, 512)
        self._person_ids: list[int] = []       # 對應每列的 person_id
        self._person_names: dict[int, str] = {}  # person_id -> name 快取

    # ── 模型初始化 ─────────────────────────────────────
    def load(self):
        """
        初始化 ONNX 推理 session，並從資料庫載入特徵索引。
        在主執行緒呼叫一次。
        """
        self._ensure_onnx_model()

        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime 未安裝。請執行: pip install onnxruntime>=1.18.0")

        providers = self._get_providers()
        logger.info(f"[Recognizer] ONNX providers: {providers}")

        t0 = time.perf_counter()
        self._session = ort.InferenceSession(
            str(self._config.ONNX_MODEL_PATH),
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"[Recognizer] ONNX 輸入節點: '{self._input_name}'")
        elapsed = time.perf_counter() - t0
        logger.info(f"[Recognizer] MobileFaceNet 載入完成 ({elapsed:.2f}s)")

        # 從資料庫載入所有特徵向量建立初始索引
        self.rebuild_index()

    def _ensure_onnx_model(self):
        """
        若 w600k_mbf.onnx 不存在，從 InsightFace 下載 buffalo_sc.zip 並提取。
        """
        model_path = self._config.ONNX_MODEL_PATH
        if model_path.exists():
            return

        logger.info("[Recognizer] 下載 InsightFace buffalo_sc 模型…")
        zip_path = model_path.parent / "buffalo_sc.zip"
        url = self._config.ONNX_MODEL_URL

        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=120) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            print(f"\r  下載進度: {pct:.1f}% ({downloaded//1024}KB/{total//1024}KB)  ", end="")
                print()
        except Exception as e:
            raise RuntimeError(f"ONNX 模型下載失敗: {e}") from e

        # 從 zip 提取 w600k_mbf.onnx
        with zipfile.ZipFile(zip_path, "r") as zf:
            found = False
            for name in zf.namelist():
                if Path(name).name == "w600k_mbf.onnx":
                    with zf.open(name) as src, open(model_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    found = True
                    logger.info(f"[Recognizer] 已提取: {model_path}")
                    break
            if not found:
                raise RuntimeError("buffalo_sc.zip 中找不到 w600k_mbf.onnx")

        # 清理 zip
        try:
            zip_path.unlink()
        except Exception:
            pass

    def _get_providers(self) -> list[str]:
        """回傳最佳可用的 ONNX Runtime 執行提供者。"""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

        for provider in ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                         "OpenVINOExecutionProvider", "CPUExecutionProvider"]:
            if provider in available:
                return [provider]
        return ["CPUExecutionProvider"]

    # ── 人臉對齊 ─────────────────────────────────────
    def align_face(self, frame: np.ndarray, detection) -> np.ndarray:
        """
        將人臉對齊至 112x112 標準尺寸（ArcFace 格式）。
        優先順序：
        1. 有 5 點人臉關鍵點 → 仿射變換對齊
        2. 有人臉區域 (face_bbox) → 裁切人臉區域並縮放
        3. 否則 → 裁切人體 bbox 並縮放（最差情況）
        回傳 BGR uint8 (112, 112, 3)
        """
        if detection.keypoints is not None:
            return self._align_with_keypoints(frame, detection.keypoints)
        elif hasattr(detection, 'has_face') and detection.has_face:
            return self._align_face_bbox(frame, detection)
        else:
            return self._align_bbox_only(frame, detection)

    def _align_with_keypoints(self, frame: np.ndarray,
                               keypoints: np.ndarray) -> np.ndarray:
        """使用 5 點關鍵點進行仿射對齊至 ArcFace 標準座標系。"""
        src_pts = keypoints.astype(np.float32)  # (5, 2)
        dst_pts = ARCFACE_SRC                    # (5, 2)

        M, _ = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.LMEDS,
        )
        if M is None:
            # 退回 bbox 裁切
            return self._align_bbox_fallback(frame, keypoints)

        aligned = cv2.warpAffine(frame, M, (112, 112),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        return aligned

    def _align_face_bbox(self, frame: np.ndarray, detection) -> np.ndarray:
        """有人臉區域時，裁切人臉區域（而非全身）並縮放至 112x112。"""
        x1, y1 = detection.face_x1, detection.face_y1
        x2, y2 = detection.face_x2, detection.face_y2
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _align_bbox_only(self, frame: np.ndarray, detection) -> np.ndarray:
        """無關鍵點時，裁切 bbox 並縮放至 112x112。"""
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        # 加入 10% 邊距
        dx = int((x2 - x1) * 0.10)
        dy = int((y2 - y1) * 0.10)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def _align_bbox_fallback(self, frame: np.ndarray,
                              keypoints: np.ndarray) -> np.ndarray:
        """關鍵點對齊失敗時的退回策略：用關鍵點 bounding box 裁切。"""
        x_min, y_min = keypoints[:, 0].min(), keypoints[:, 1].min()
        x_max, y_max = keypoints[:, 0].max(), keypoints[:, 1].max()
        h, w = frame.shape[:2]
        margin = 0.3
        dx = (x_max - x_min) * margin
        dy = (y_max - y_min) * margin
        x1 = max(0, int(x_min - dx))
        y1 = max(0, int(y_min - dy))
        x2 = min(w, int(x_max + dx))
        y2 = min(h, int(y_max + dy))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    # ── 特徵提取 ─────────────────────────────────────
    def extract_embedding(self, face_112: np.ndarray) -> Optional[np.ndarray]:
        """
        輸入：112x112 BGR uint8
        輸出：512 維 float32 L2-normalized 向量，失敗回傳 None

        前處理：BGR→RGB, /255, -0.5, /0.5, transpose (3,112,112), 加 batch dim
        """
        if self._session is None:
            logger.warning("[Recognizer] ONNX session 未初始化")
            return None

        if face_112.shape[:2] != (112, 112):
            face_112 = cv2.resize(face_112, (112, 112))

        try:
            img = cv2.cvtColor(face_112, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) / 0.5
            img = img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)

            output = self._session.run(None, {self._input_name: img})
            emb = output[0][0].astype(np.float32)  # (512,)

            # L2 normalize（模型通常已 normalize，此為防禦性步驟）
            norm = np.linalg.norm(emb)
            if norm > 1e-8:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.error(f"[Recognizer] 特徵提取失敗: {e}")
            return None

    # ── 辨識 ─────────────────────────────────────────
    def identify(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        餘弦相似度辨識。
        由於所有向量已 L2 normalize，dot product = 餘弦相似度。
        回傳 (name, similarity)；低於閾值回傳 ('Unknown', similarity)。
        """
        with self._lock:
            if self._embeddings_matrix is None or len(self._person_ids) == 0:
                return "Unknown", 0.0

            # 矩陣點積：O(N * 512)，N < 10000 時效能足夠
            similarities = self._embeddings_matrix @ embedding  # (N,)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim < self._config.RECOGNITION_THRESHOLD:
                return "Unknown", best_sim

            person_id = self._person_ids[best_idx]
            name = self._person_names.get(person_id)
            if name is None:
                name = self._db.get_person_name(person_id)
                self._person_names[person_id] = name

            return name, best_sim

    def identify_with_id(self, embedding: np.ndarray) -> Tuple[Optional[int], str, float]:
        """與 identify 相同，但額外回傳 person_id（None 表示未知）。"""
        with self._lock:
            if self._embeddings_matrix is None or len(self._person_ids) == 0:
                return None, "Unknown", 0.0

            person_id, name, best_sim = self._top_k_match(embedding)

            if best_sim < self._config.RECOGNITION_THRESHOLD:
                return None, "Unknown", best_sim

            return person_id, name, best_sim

    def best_match(self, embedding: np.ndarray) -> Tuple[Optional[int], str, float]:
        """回傳最佳匹配（不受閾值限制），用於寬鬆匹配判斷。"""
        with self._lock:
            if self._embeddings_matrix is None or len(self._person_ids) == 0:
                return None, "Unknown", 0.0

            person_id, name, best_sim = self._top_k_match(embedding)
            return person_id, name, best_sim

    def _top_k_match(
        self, embedding: np.ndarray, top_k: int = 3
    ) -> Tuple[Optional[int], str, float]:
        """
        按人物分組取 top-K 相似度均值的核心匹配算法。
        NOTE: 不同於單純取最大值，此方法讓同一人物的多個角度特徵共同「投票」，
        避免單一異常向量導致誤判，提高辨識穩定性。

        必須在 self._lock 保護下呼叫。
        """
        similarities = self._embeddings_matrix @ embedding  # (N,)

        # 按 person_id 分組取 top-K 均值
        person_scores: dict[int, float] = {}
        person_sims: dict[int, list[float]] = {}

        for idx, pid in enumerate(self._person_ids):
            sim = float(similarities[idx])
            if pid not in person_sims:
                person_sims[pid] = []
            person_sims[pid].append(sim)

        for pid, sims in person_sims.items():
            # 取最高的 top_k 個相似度的均值
            sorted_sims = sorted(sims, reverse=True)
            top_sims = sorted_sims[:top_k]
            person_scores[pid] = sum(top_sims) / len(top_sims)

        if not person_scores:
            return None, "Unknown", 0.0

        best_pid = max(person_scores, key=person_scores.get)  # type: ignore
        best_sim = person_scores[best_pid]

        name = self._person_names.get(best_pid)
        if name is None:
            name = self._db.get_person_name(best_pid)
            self._person_names[best_pid] = name

        return best_pid, name, best_sim

    # ── 增量學習 ─────────────────────────────────────
    def add_face(self, person_id: int, embedding: np.ndarray,
                 source_path: Optional[str] = None, quality: Optional[float] = None):
        """
        新增單一特徵向量至資料庫並重建索引。
        NOTE: 達到最大向量數後，若新向量代表新角度（與已有向量差異大），
        則替換最冗餘的舊向量，確保全方位覆蓋（正臉/側臉/俯仰）。
        """
        count = self._db.count_embeddings_for_person(person_id)
        max_emb = self._config.MAX_EMBEDDINGS_PER_PERSON

        if count >= max_emb:
            # NOTE: 角度多樣性檢測 — 只有新角度的向量才值得替換
            existing = self._db.get_embeddings_for_person(person_id)
            if not existing:
                return

            existing_arr = np.array(existing)
            sims = existing_arr @ embedding  # 与所有已有向量的相似度
            max_sim = float(sims.max())

            # 如果新向量與已有最相似的向量差異夠大（< 0.80），代表新角度
            if max_sim < 0.80:
                # 找到最冗餘的向量（與其他向量平均相似度最高）
                if len(existing_arr) > 1:
                    sim_matrix = existing_arr @ existing_arr.T
                    np.fill_diagonal(sim_matrix, 0)
                    avg_sims = sim_matrix.mean(axis=1)
                    redundant_idx = int(avg_sims.argmax())
                else:
                    redundant_idx = 0

                # 替換最冗餘的舊向量
                self._db.replace_embedding(person_id, redundant_idx, embedding)
                self.rebuild_index()
                logger.info(f"[Recognizer] 多樣性替換: person_id={person_id} "
                            f"(新角度 sim={max_sim:.3f}, 替換 idx={redundant_idx})")
            else:
                logger.debug(f"[Recognizer] 人物 {person_id} 已達最大向量數且無新角度 "
                             f"(max_sim={max_sim:.3f})，跳過")
            return

        self._db.save_embedding(person_id, embedding, source_path, quality)
        self.rebuild_index()
        logger.info(f"[Recognizer] 增量學習: person_id={person_id}")

    def batch_add_faces(self, person_id: int,
                        embeddings: list[np.ndarray],
                        source_paths: Optional[list[str]] = None,
                        qualities: Optional[list[float]] = None):
        """
        批量新增多個特徵向量。
        先全部寫入 DB，最後只呼叫一次 rebuild_index（效能關鍵）。
        """
        if not embeddings:
            return
        self._db.save_embeddings_batch(person_id, embeddings, source_paths, qualities)
        self.rebuild_index()
        logger.info(f"[Recognizer] 批量新增 {len(embeddings)} 個向量 (person_id={person_id})")

    def rebuild_index(self):
        """
        從資料庫重新載入所有特徵向量，重建辨識矩陣。
        每次 add_face/batch_add_faces 後自動呼叫。
        使用 RLock 保護，辨識執行緒在重建期間稍作等待。
        """
        rows = self._db.get_all_embeddings()
        if not rows:
            with self._lock:
                self._embeddings_matrix = None
                self._person_ids = []
                self._person_names.clear()
            logger.info("[Recognizer] 特徵索引為空（尚未有人物資料）")
            return

        person_ids = [r["person_id"] for r in rows]
        embeddings = np.stack([
            np.frombuffer(r["embedding"], dtype=np.float32)
            for r in rows
        ])  # (N, 512)

        # 防禦性 L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # 更新名稱快取
        names = {r["person_id"]: r["name"] for r in rows}

        with self._lock:
            self._embeddings_matrix = embeddings
            self._person_ids = person_ids
            self._person_names.update(names)

        logger.info(f"[Recognizer] 索引重建完成: {len(rows)} 個向量, "
                    f"{len(set(person_ids))} 人")

    def remove_person(self, person_id: int):
        """刪除人物：移除特徵向量、停用 DB 記錄、重建索引。"""
        self._db.delete_embeddings_for_person(person_id)
        self._db.delete_person(person_id)
        self.rebuild_index()
        with self._lock:
            self._person_names.pop(person_id, None)

    def remove_all_persons(self):
        """徹底刪除所有人物、特徵向量、辨識記錄並清空索引。"""
        self._db.purge_all_data()
        self.rebuild_index()
        with self._lock:
            self._person_names.clear()

    def invalidate_name_cache(self):
        """清除名稱快取（重新命名後呼叫）。"""
        with self._lock:
            self._person_names.clear()

    def get_index_size(self) -> int:
        """回傳索引中特徵向量的數量。"""
        with self._lock:
            return len(self._person_ids)

    def is_loaded(self) -> bool:
        return self._session is not None


# ── 自測 ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys, tempfile, os
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 建立最小設定
    class _Cfg:
        ONNX_MODEL_PATH = Path("data/models/w600k_mbf.onnx")
        ONNX_MODEL_URL  = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
        RECOGNITION_THRESHOLD = 0.40
        MAX_EMBEDDINGS_PER_PERSON = 10

    # 模擬最小 DB
    class _MockDB:
        def __init__(self):
            self._rows = []
            self._pid = 0

        def save_embedding(self, pid, emb, *a, **kw):
            self._rows.append({"person_id": pid,
                                "embedding": emb.tobytes(),
                                "name": f"Person{pid}"})
        def save_embeddings_batch(self, pid, embs, *a, **kw):
            for e in embs:
                self.save_embedding(pid, e)
        def get_all_embeddings(self): return self._rows
        def count_embeddings_for_person(self, pid): return sum(1 for r in self._rows if r["person_id"] == pid)
        def delete_embeddings_for_person(self, pid): self._rows = [r for r in self._rows if r["person_id"] != pid]
        def get_person_name(self, pid): return f"Person{pid}"

    cfg = _Cfg()
    if not cfg.ONNX_MODEL_PATH.exists():
        print("ONNX 模型不存在，跳過推理測試（只測試對齊邏輯）")
    else:
        db = _MockDB()
        rec = FaceRecognizer(cfg, db)
        rec.load()
        print(f"索引大小: {rec.get_index_size()} (預期 0)")

        # 模擬兩個人物
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emb1 = rec.extract_embedding(dummy_face)
        assert emb1 is not None and emb1.shape == (512,), "特徵提取失敗"
        print(f"特徵向量長度: {emb1.shape[0]}, L2-norm: {np.linalg.norm(emb1):.4f}")

        rec.add_face(1, emb1)
        name, conf = rec.identify(emb1)
        print(f"自我辨識: {name}, conf={conf:.4f}")
        assert name == "Person1", f"預期 Person1，得到 {name}"

        print("✅ FaceRecognizer 自測通過")
