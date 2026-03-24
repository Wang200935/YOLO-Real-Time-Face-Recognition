"""
main.py — 程式入口
初始化順序：
  1. 日誌設定
  2. 建立目錄結構
  3. 初始化資料庫
  4. 載入 YOLOv11 檢測模型
  5. 載入 MobileFaceNet 特徵提取模型
  6. 建立佇列與執行緒管理器
  7. 啟動 Tkinter 主視窗
"""

import logging
import queue
import sys
import threading
import time
from pathlib import Path


def setup_logging():
    """設定 logging，輸出至終端機與日誌檔。"""
    log_dir = Path(__file__).parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    # 降低 ultralytics 和 PIL 的日誌級別（太吵）
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def check_dependencies():
    """檢查必要套件，提早回報缺失項目。"""
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "PIL": "Pillow",
        "onnxruntime": "onnxruntime>=1.18.0",
        "ultralytics": "ultralytics>=8.3.0",
        "pygame": "pygame",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ 缺少以下套件，請執行安裝指令：")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)

    # 檢查 ultralytics 版本
    try:
        import ultralytics
        ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
        if ver < (8, 3):
            print(f"⚠  ultralytics 版本 {ultralytics.__version__} 過低，"
                  f"YOLOv11 需要 >= 8.3.0")
            print("   請執行: pip install ultralytics>=8.3.0")
            sys.exit(1)
    except Exception:
        pass


def main():
    setup_logging()
    logger = logging.getLogger("main")

    print("=" * 60)
    print(" YOLOv11 即時人臉辨識系統")
    print("=" * 60)

    # ── 1. 依賴檢查 ─────────────────────────────────
    print("[1/5] 檢查依賴套件…")
    check_dependencies()
    print("      ✅ 依賴套件完整")

    # ── 2. 設定載入 ─────────────────────────────────
    from config import Config, DATA_DIR
    config = Config()
    logger.info(f"資料目錄: {DATA_DIR}")

    # ── 3. 資料庫初始化 ──────────────────────────────
    print("[2/5] 初始化資料庫…")
    from database import DatabaseManager
    db = DatabaseManager(config.DB_PATH)
    stats = db.get_stats()
    print(f"      ✅ 資料庫就緒 ({stats['persons']} 人, {stats['embeddings']} 個向量)")

    # ── 4. 載入 YOLOv11 模型 ───────────────────────────
    print("[3/5] 載入 YOLOv11 人臉檢測模型…")
    from face_detector import FaceDetector
    detector = FaceDetector(config)
    try:
        detector.load()
        print("      ✅ YOLOv11 模型載入成功")
    except Exception as e:
        print(f"      ❌ YOLOv11 模型載入失敗: {e}")
        logger.exception("YOLOv11 載入失敗")
        input("按 Enter 退出…")
        sys.exit(1)

    # ── 5. 載入 MobileFaceNet ONNX 模型 ────────────────
    print("[4/5] 載入 MobileFaceNet 特徵提取模型…")
    from face_recognizer import FaceRecognizer
    recognizer = FaceRecognizer(config, db)
    try:
        recognizer.load()
        idx_size = recognizer.get_index_size()
        print(f"      ✅ MobileFaceNet 載入成功（索引: {idx_size} 個向量）")
    except Exception as e:
        print(f"      ❌ MobileFaceNet 載入失敗: {e}")
        logger.exception("MobileFaceNet 載入失敗")
        input("按 Enter 退出…")
        sys.exit(1)

    # ── 6. 建立多執行緒架構 ───────────────────────────
    print("[5/5] 初始化多執行緒架構…")
    from utils import AlertManager
    from worker_threads import ThreadManager

    alert_manager = AlertManager(
        cooldown_s=config.ALERT_COOLDOWN_S,
        sound_path=config.ALERT_SOUND_PATH,
        enable_sound=config.ENABLE_SOUND,
    )

    result_queue = queue.Queue(maxsize=config.RESULT_QUEUE_SIZE)
    alert_names: set[str] = set()
    alert_lock  = threading.Lock()

    thread_manager = ThreadManager(
        config=config,
        detector=detector,
        recognizer=recognizer,
        db_manager=db,
        result_queue=result_queue,
        alert_names=alert_names,
        alert_lock=alert_lock,
    )
    print("      ✅ 執行緒架構就緒")

    # ── 7. 啟動 GUI ──────────────────────────────────
    print("\n啟動圖形介面…\n")
    from gui import MainWindow

    try:
        app = MainWindow(
            config=config,
            recognizer=recognizer,
            db_manager=db,
            result_queue=result_queue,
            thread_manager=thread_manager,
            alert_manager=alert_manager,
            alert_names=alert_names,
            alert_lock=alert_lock,
        )
        app.run()  # 阻塞直到視窗關閉
    except KeyboardInterrupt:
        print("\n使用者中斷，正在關閉…")
    except Exception as e:
        logger.exception("GUI 發生未預期錯誤")
        print(f"❌ GUI 錯誤: {e}")
    finally:
        if thread_manager.is_running():
            thread_manager.stop_all(timeout=3.0)
        db.close()
        print("系統已關閉。")


if __name__ == "__main__":
    main()
