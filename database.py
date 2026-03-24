"""
database.py — SQLite 資料管理
- WAL 模式支援多執行緒並發讀寫
- threading.local 每執行緒獨立連線，避免 "database is locked" 錯誤
- Embedding 使用 np.float32.tobytes() 序列化（非 pickle）
"""

import sqlite3
import threading
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite 資料庫封裝。
    每個執行緒透過 threading.local 維護各自獨立的 Connection，
    避免跨執行緒共享連線導致的執行緒安全問題。
    """

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS persons (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        employee_id TEXT UNIQUE,
        department  TEXT,
        notes       TEXT,
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL,
        thumbnail   BLOB,
        is_active   INTEGER NOT NULL DEFAULT 1
    );

    CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);

    CREATE TABLE IF NOT EXISTS face_embeddings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
        embedding   BLOB NOT NULL,
        source_path TEXT,
        quality     REAL,
        created_at  REAL NOT NULL,
        updated_at  REAL NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_person_id ON face_embeddings(person_id);

    CREATE TABLE IF NOT EXISTS recognition_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id   INTEGER REFERENCES persons(id) ON DELETE SET NULL,
        confidence  REAL NOT NULL,
        timestamp   REAL NOT NULL,
        is_unknown  INTEGER NOT NULL DEFAULT 0,
        frame_path  TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_log_timestamp ON recognition_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_log_person_id ON recognition_log(person_id);
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # 在主执行绪初始化 schema
        self.initialize_schema()
        # NOTE: 启动时执行 ANALYZE 优化查询计划
        self._run_analyze()

    # ── 連線管理 ───────────────────────────────────────
    @property
    def _conn(self) -> sqlite3.Connection:
        """
        懶初始化：每個執行緒首次存取時建立自己的連線。
        啟用 WAL 模式、外鍵約束與正常同步模式。
        """
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                timeout=15.0,
                check_same_thread=False,  # 安全：每執行緒獨立 connection
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB cache
            self._local.conn = conn
            logger.debug(f"[DB] 新連線建立於執行緒 {threading.current_thread().name}")
        return self._local.conn

    def close(self):
        """關閉當前執行緒的連線。"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def initialize_schema(self):
        """建立資料表（若不存在）。"""
        try:
            self._conn.executescript(self.SCHEMA_SQL)
            self._conn.commit()
            logger.info("[DB] Schema 初始化完成")
        except sqlite3.Error as e:
            logger.error(f"[DB] Schema 初始化失敗: {e}")
            raise

    # ── persons CRUD ────────────────────────────────────
    def add_person(
        self,
        name: str,
        employee_id: Optional[str] = None,
        department: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """新增人物，回傳新 person_id。"""
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO persons (name, employee_id, department, notes,
               created_at, updated_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, 1)""",
            (name, employee_id, department, notes, now, now),
        )
        self._conn.commit()
        pid = cur.lastrowid
        logger.info(f"[DB] 新增人物: {name} (id={pid})")
        return pid

    def get_person(self, person_id: int) -> Optional[dict]:
        """依 ID 查詢人物，回傳 dict 或 None。"""
        row = self._conn.execute(
            "SELECT * FROM persons WHERE id=? AND is_active=1", (person_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """依名稱查詢人物（精確比對）。"""
        row = self._conn.execute(
            "SELECT * FROM persons WHERE name=? AND is_active=1", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_persons(self) -> list[dict]:
        """取得所有啟用中的人物清單。"""
        rows = self._conn.execute(
            "SELECT * FROM persons WHERE is_active=1 ORDER BY name ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_person(self, person_id: int, **kwargs) -> bool:
        """
        更新人物欄位（name, employee_id, department, notes）。
        回傳 True 表示有更新。
        """
        allowed = {"name", "employee_id", "department", "notes", "thumbnail"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [person_id]
        self._conn.execute(f"UPDATE persons SET {cols} WHERE id=?", vals)
        self._conn.commit()
        return True

    def rename_person(self, person_id: int, new_name: str) -> bool:
        """重新命名人物，並檢查名稱唯一性。回傳 True 表示成功。"""
        existing = self.get_person_by_name(new_name)
        if existing and existing["id"] != person_id:
            logger.warning(f"[DB] 名稱已存在: {new_name}")
            return False
        self.update_person(person_id, name=new_name)
        logger.info(f"[DB] 人物 {person_id} 更名為: {new_name}")
        return True

    def delete_person(self, person_id: int) -> bool:
        """硬刪除人物及其所有特徵向量（永久移除）。"""
        self._conn.execute(
            "DELETE FROM face_embeddings WHERE person_id=?", (person_id,)
        )
        self._conn.execute(
            "DELETE FROM persons WHERE id=?", (person_id,)
        )
        self._conn.commit()
        logger.info(f"[DB] 硬刪除人物 id={person_id}")
        return True

    def set_thumbnail(self, person_id: int, jpeg_bytes: bytes):
        """儲存人物縮圖（JPEG bytes）。"""
        self._conn.execute(
            "UPDATE persons SET thumbnail=?, updated_at=? WHERE id=?",
            (jpeg_bytes, time.time(), person_id),
        )
        self._conn.commit()

    def person_exists(self, name: str) -> bool:
        """檢查名稱是否已存在。"""
        row = self._conn.execute(
            "SELECT id FROM persons WHERE name=? AND is_active=1", (name,)
        ).fetchone()
        return row is not None

    def get_next_visitor_name(self) -> str:
        """
        取得下一個可用的「訪客N」名稱。
        自動掃描現有訪客編號，回傳 max+1。
        """
        rows = self._conn.execute(
            "SELECT name FROM persons WHERE name LIKE '訪客%' AND is_active=1"
        ).fetchall()
        nums = []
        for r in rows:
            name = r["name"]
            suffix = name[2:]  # 去掉「訪客」
            if suffix.isdigit():
                nums.append(int(suffix))
        n = max(nums) + 1 if nums else 1
        return f"訪客{n}"

    # ── 特徵向量 CRUD ────────────────────────────────────
    def save_embedding(
        self,
        person_id: int,
        embedding: np.ndarray,
        source_path: Optional[str] = None,
        quality: Optional[float] = None,
    ) -> int:
        """儲存特徵向量，回傳新 embedding_id。"""
        blob = embedding.astype(np.float32).tobytes()
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO face_embeddings (person_id, embedding, source_path,
               quality, created_at) VALUES (?, ?, ?, ?, ?)""",
            (person_id, blob, source_path, quality, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def save_embeddings_batch(
        self,
        person_id: int,
        embeddings: list[np.ndarray],
        source_paths: Optional[list[str]] = None,
        qualities: Optional[list[float]] = None,
    ):
        """批量儲存多個特徵向量（一次 commit，提升效能）。"""
        now = time.time()
        rows = []
        for i, emb in enumerate(embeddings):
            blob = emb.astype(np.float32).tobytes()
            src = source_paths[i] if source_paths and i < len(source_paths) else None
            q   = qualities[i]    if qualities and i < len(qualities)         else None
            rows.append((person_id, blob, src, q, now))
        self._conn.executemany(
            """INSERT INTO face_embeddings (person_id, embedding, source_path,
               quality, created_at) VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        logger.info(f"[DB] 批量儲存 {len(embeddings)} 個特徵向量 (person_id={person_id})")

    def get_all_embeddings(self) -> list[dict]:
        """
        讀取所有啟用人物的特徵向量（含 person_id、embedding blob）。
        用於重建辨識索引。
        """
        rows = self._conn.execute(
            """SELECT fe.id, fe.person_id, fe.embedding, p.name
               FROM face_embeddings fe
               JOIN persons p ON fe.person_id = p.id
               WHERE p.is_active = 1
               ORDER BY fe.person_id, fe.id"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_embeddings_for_person(self, person_id: int) -> list[np.ndarray]:
        """取得特定人物的所有特徵向量（numpy arrays）。"""
        rows = self._conn.execute(
            "SELECT embedding FROM face_embeddings WHERE person_id=?", (person_id,)
        ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]

    def count_embeddings_for_person(self, person_id: int) -> int:
        """計算特定人物已存的特徵向量數量。"""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM face_embeddings WHERE person_id=?", (person_id,)
        ).fetchone()
        return row["cnt"] if row else 0

    def replace_embedding(self, person_id: int, idx: int, new_embedding: np.ndarray):
        """
        替換指定人物的第 idx 條特徵向量（用於角度多樣性替換）。
        NOTE: 用 rowid as rid 別名，避免 SQLite Row 無法直接存取 rowid。
        """
        rows = self._conn.execute(
            "SELECT rowid as rid FROM face_embeddings WHERE person_id=? ORDER BY rowid",
            (person_id,)
        ).fetchall()
        if idx < len(rows):
            target_rowid = rows[idx]["rid"]
            blob = new_embedding.astype(np.float32).tobytes()
            self._conn.execute(
                "UPDATE face_embeddings SET embedding=?, updated_at=? "
                "WHERE rowid=?",
                (blob, time.time(), target_rowid)
            )
            self._conn.commit()
            logger.debug(f"[DB] 替換 person_id={person_id} 的第 {idx} 條向量")

    def delete_embeddings_for_person(self, person_id: int):
        """刪除特定人物的所有特徵向量。"""
        self._conn.execute(
            "DELETE FROM face_embeddings WHERE person_id=?", (person_id,)
        )
        self._conn.commit()

    # ── 辨識記錄 ────────────────────────────────────────
    def log_recognition(
        self,
        person_id: Optional[int],
        confidence: float,
        is_unknown: bool = False,
        frame_path: Optional[str] = None,
    ):
        """記錄辨識結果（由 DBWriterThread 非同步呼叫）。"""
        self._conn.execute(
            """INSERT INTO recognition_log (person_id, confidence, timestamp,
               is_unknown, frame_path) VALUES (?, ?, ?, ?, ?)""",
            (person_id, confidence, time.time(), int(is_unknown), frame_path),
        )
        self._conn.commit()

    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        """取得最近 N 筆辨識記錄。"""
        rows = self._conn.execute(
            """SELECT rl.*, p.name
               FROM recognition_log rl
               LEFT JOIN persons p ON rl.person_id = p.id
               ORDER BY rl.timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_recognition_log(self):
        """清空辨識記錄表。"""
        self._conn.execute("DELETE FROM recognition_log")
        self._conn.commit()
        logger.info("[DB] 已清空辨識記錄")

    def purge_all_data(self):
        """徹底清除所有資料（人物、特徵向量、辨識記錄），包含歷史殘留。"""
        self._conn.execute("DELETE FROM recognition_log")
        self._conn.execute("DELETE FROM face_embeddings")
        self._conn.execute("DELETE FROM persons")
        self._conn.commit()
        logger.info("[DB] 已徹底清除所有資料")

    def log_recognition_batch(self, records: list[dict]):
        """批量写入辨识记录，减少 I/O 次数。"""
        if not records:
            return
        now = time.time()
        rows = [
            (r.get("person_id"), r["confidence"], now,
             int(r.get("is_unknown", False)), r.get("frame_path"))
            for r in records
        ]
        self._conn.executemany(
            """INSERT INTO recognition_log (person_id, confidence, timestamp,
               is_unknown, frame_path) VALUES (?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def optimize(self):
        """执行 VACUUM 和 ANALYZE 优化数据库性能。"""
        try:
            self._conn.execute("VACUUM")
            self._conn.execute("ANALYZE")
            logger.info("[DB] 数据库优化完成 (VACUUM + ANALYZE)")
        except sqlite3.Error as e:
            logger.error(f"[DB] 数据库优化失败: {e}")

    def _run_analyze(self):
        """启动时执行 ANALYZE，更新统计信息以优化查询计划。"""
        try:
            self._conn.execute("ANALYZE")
        except Exception:
            pass

    # ── 統計 ────────────────────────────────────────────
    def get_stats(self) -> dict:
        """回傳資料庫統計資訊（只計算啟用中人物的數據）。"""
        persons_count = self._conn.execute(
            "SELECT COUNT(*) as c FROM persons WHERE is_active=1"
        ).fetchone()["c"]
        embeddings_count = self._conn.execute(
            """SELECT COUNT(*) as c FROM face_embeddings fe
               JOIN persons p ON fe.person_id = p.id
               WHERE p.is_active = 1"""
        ).fetchone()["c"]
        return {
            "persons": persons_count,
            "embeddings": embeddings_count,
        }

    def get_person_name(self, person_id: int) -> str:
        """快速取得人物名稱，找不到回傳 'Unknown'。"""
        row = self._conn.execute(
            "SELECT name FROM persons WHERE id=? AND is_active=1", (person_id,)
        ).fetchone()
        return row["name"] if row else "Unknown"


# ── 自測 ────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pathlib import Path
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        db = DatabaseManager(db_path)

        # 測試新增人物
        pid = db.add_person("測試者A", notes="自測")
        assert pid == 1, "person_id 應為 1"

        # 測試特徵向量
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        eid = db.save_embedding(pid, emb, quality=0.95)
        assert eid == 1, "embedding_id 應為 1"

        # 測試讀取
        rows = db.get_all_embeddings()
        assert len(rows) == 1
        loaded = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        assert np.allclose(emb, loaded), "特徵向量讀寫不一致"

        # 測試統計
        stats = db.get_stats()
        assert stats["persons"] == 1
        assert stats["embeddings"] == 1

        # 測試訪客名稱
        name = db.get_next_visitor_name()
        assert name == "訪客1", f"預期訪客1，得到 {name}"

        # 測試刪除
        db.delete_person(pid)
        assert db.get_person(pid) is None

        print("✅ DatabaseManager 自測通過")
