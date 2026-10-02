"""Storage: LanceDB (chunks + vectors + FTS) and SQLite (manifest, persistent queue, meta)."""
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import indexer_config as cfg

COLUMNS = ["text", "path", "project", "kind", "source", "ext", "symbol",
           "start_line", "end_line", "page", "chunk_hash", "model_id", "mtime"]


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _chunked(seq: Sequence, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def make_schema(dim: int):
    import pyarrow as pa
    return pa.schema([
        ("vector", pa.list_(pa.float32(), dim)),
        ("text", pa.string()),
        ("path", pa.string()),
        ("project", pa.string()),
        ("kind", pa.string()),
        ("source", pa.string()),
        ("ext", pa.string()),
        ("symbol", pa.string()),
        ("start_line", pa.int32()),
        ("end_line", pa.int32()),
        ("page", pa.int32()),
        ("chunk_hash", pa.string()),
        ("model_id", pa.string()),
        ("mtime", pa.int64()),
    ])


class State:
    """SQLite part: manifest, persistent queue, meta. Light enough for the always-on watcher."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir or cfg.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.sql = sqlite3.connect(str(self.data_dir / "state.sqlite"), timeout=30,
                                   check_same_thread=False, isolation_level=None)
        self.sql.execute("PRAGMA journal_mode=WAL")
        self.sql.execute("PRAGMA synchronous=NORMAL")
        self.sql.executescript("""
            CREATE TABLE IF NOT EXISTS manifest(
                path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER,
                content_hash TEXT, indexed_at REAL);
            CREATE TABLE IF NOT EXISTS queue(
                path TEXT PRIMARY KEY, op TEXT NOT NULL, priority REAL DEFAULT 0,
                not_before REAL DEFAULT 0, seq INTEGER NOT NULL, attempts INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        """)

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self.sql.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self.sql.execute("INSERT INTO meta(key,value) VALUES(?,?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    # -------------------------------------------------------------- manifest
    def manifest_get(self, path: str) -> Optional[Tuple[int, int, str]]:
        with self._lock:
            row = self.sql.execute("SELECT mtime_ns,size,content_hash FROM manifest WHERE path=?",
                                   (path,)).fetchone()
        return tuple(row) if row else None

    def manifest_all(self) -> Dict[str, Tuple[int, int]]:
        with self._lock:
            rows = self.sql.execute("SELECT path,mtime_ns,size FROM manifest").fetchall()
        return {p: (m, s) for p, m, s in rows}

    def manifest_set(self, path: str, mtime_ns: int, size: int, content_hash: str) -> None:
        with self._lock:
            self.sql.execute(
                "INSERT INTO manifest(path,mtime_ns,size,content_hash,indexed_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET mtime_ns=excluded.mtime_ns,size=excluded.size,"
                "content_hash=excluded.content_hash,indexed_at=excluded.indexed_at",
                (path, mtime_ns, size, content_hash, time.time()))

    def manifest_delete(self, path: str) -> None:
        with self._lock:
            self.sql.execute("DELETE FROM manifest WHERE path=?", (path,))

    def manifest_under(self, prefix: str) -> List[str]:
        """Manifest paths inside a directory (case-insensitive)."""
        prefix = prefix.rstrip("\\/") + os.sep
        with self._lock:
            rows = self.sql.execute("SELECT path FROM manifest WHERE substr(path,1,?) = ? COLLATE NOCASE",
                                    (len(prefix), prefix)).fetchall()
        return [r[0] for r in rows]

    def manifest_count(self) -> int:
        with self._lock:
            return self.sql.execute("SELECT COUNT(*) FROM manifest").fetchone()[0]

    # ----------------------------------------------------------------- queue
    def enqueue(self, path: str, op: str = "upsert", priority: float = 0.0,
                delay: float = 0.0) -> None:
        now = time.time()
        with self._lock:
            self.sql.execute(
                "INSERT INTO queue(path,op,priority,not_before,seq,attempts) VALUES(?,?,?,?,?,0) "
                "ON CONFLICT(path) DO UPDATE SET op=excluded.op,priority=excluded.priority,"
                "not_before=excluded.not_before,seq=excluded.seq,attempts=0",
                (path, op, priority, now + delay, time.time_ns()))

    def enqueue_many(self, items: Iterable[Tuple[str, str, float]]) -> int:
        """items: (path, op, priority). Single transaction; used by reconcile / initial index."""
        n = 0
        now = time.time()
        with self._lock:
            self.sql.execute("BEGIN")
            try:
                for path, op, priority in items:
                    self.sql.execute(
                        "INSERT INTO queue(path,op,priority,not_before,seq,attempts) VALUES(?,?,?,?,?,0) "
                        "ON CONFLICT(path) DO UPDATE SET op=excluded.op,priority=excluded.priority,"
                        "not_before=excluded.not_before,seq=excluded.seq,attempts=0",
                        (path, op, priority, now, time.time_ns()))
                    n += 1
                self.sql.execute("COMMIT")
            except Exception:
                self.sql.execute("ROLLBACK")
                raise
        return n

    def queue_size(self, due_only: bool = False) -> int:
        with self._lock:
            if due_only:
                return self.sql.execute("SELECT COUNT(*) FROM queue WHERE not_before<=?",
                                        (time.time(),)).fetchone()[0]
            return self.sql.execute("SELECT COUNT(*) FROM queue").fetchone()[0]

    def claim(self, limit: int, ignore_debounce: bool = False) -> List[Tuple[str, str, int]]:
        """Highest priority first (recently modified files first). Rows stay queued until done()."""
        cutoff = 1e18 if ignore_debounce else time.time()
        with self._lock:
            rows = self.sql.execute(
                "SELECT path,op,seq FROM queue WHERE not_before<=? AND attempts<3 "
                "ORDER BY priority DESC LIMIT ?", (cutoff, limit)).fetchall()
        return [tuple(r) for r in rows]

    def done(self, items: Iterable[Tuple[str, int]]) -> None:
        """Remove processed queue rows, unless the file was re-queued (seq changed) meanwhile."""
        with self._lock:
            self.sql.execute("BEGIN")
            for path, seq in items:
                self.sql.execute("DELETE FROM queue WHERE path=? AND seq=?", (path, seq))
            self.sql.execute("COMMIT")

    def fail(self, path: str, delay: float = 300.0) -> None:
        with self._lock:
            self.sql.execute("UPDATE queue SET attempts=attempts+1, not_before=? WHERE path=?",
                             (time.time() + delay, path))

    def close(self) -> None:
        with self._lock:
            self.sql.close()


class Store(State):
    """State + the LanceDB chunk table (vectors, text, FTS)."""

    def __init__(self, data_dir: Path = None, model_id: str = None, dim: int = None,
                 read_only: bool = False):
        super().__init__(data_dir)
        import lancedb
        self.model_id = model_id or cfg.EMBED_MODEL
        self.dim = dim
        self.db = lancedb.connect(str(self.data_dir / "lance"))
        self._table = None
        if dim is not None and not read_only:
            self.check_model()
        elif read_only:
            self._open_existing()

    # ------------------------------------------------------------ lance table
    def _table_names(self) -> List[str]:
        res = self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        return list(getattr(res, "tables", res))

    def _open_existing(self) -> None:
        if cfg.LANCE_TABLE in self._table_names():
            self._table = self.db.open_table(cfg.LANCE_TABLE)

    def check_model(self) -> bool:
        """Create the table, or wipe everything if the model/dim changed. True if a wipe happened."""
        wiped = False
        stored = (self.get_meta("model_id"), self.get_meta("dim"))
        want = (self.model_id, str(self.dim))
        exists = cfg.LANCE_TABLE in self._table_names()
        if exists and stored != want:
            self.db.drop_table(cfg.LANCE_TABLE)
            with self._lock:
                self.sql.execute("DELETE FROM manifest")
            exists, wiped = False, True
        if not exists:
            self._table = self.db.create_table(cfg.LANCE_TABLE, schema=make_schema(self.dim))
            self._ensure_fts()
        else:
            self._table = self.db.open_table(cfg.LANCE_TABLE)
        self.set_meta("model_id", self.model_id)
        self.set_meta("dim", str(self.dim))
        return wiped

    @property
    def table(self):
        if self._table is None:
            self._open_existing()
        return self._table

    def _ensure_fts(self) -> None:
        from lancedb.index import FTS
        try:
            cfgs = [
                dict(stem=False, remove_stop_words=False, with_position=True, split_identifiers=True),
                dict(stem=False, remove_stop_words=False, with_position=True),
                {},
            ]
            for kw in cfgs:
                try:
                    self._table.create_index("text", config=FTS(**kw), replace=True)
                    return
                except TypeError:
                    continue
        except Exception:
            pass  # FTS is an optimisation; vector search still works

    def count(self) -> int:
        t = self.table
        return t.count_rows() if t is not None else 0

    def existing_hashes(self, path: str) -> set:
        t = self.table
        if t is None:
            return set()
        rows = t.search().where(f"path = {_q(path)}").select(["chunk_hash"]).limit(100000).to_list()
        return {r["chunk_hash"] for r in rows}

    def vectors_for_hashes(self, hashes: Iterable[str]) -> Dict[str, object]:
        """Reuse embeddings already in the table (same chunk text anywhere -> embedded once)."""
        import numpy as np
        t = self.table
        out: Dict[str, np.ndarray] = {}
        if t is None:
            return out
        hashes = list(set(hashes))
        for part in _chunked(hashes, 200):
            where = "chunk_hash IN (" + ",".join(_q(h) for h in part) + f") AND model_id = {_q(self.model_id)}"
            rows = t.search().where(where).select(["chunk_hash", "vector"]).limit(len(part) * 4).to_list()
            for r in rows:
                out.setdefault(r["chunk_hash"], np.asarray(r["vector"], dtype=np.float32))
        return out

    def delete_paths(self, paths: Iterable[str]) -> None:
        t = self.table
        if t is None:
            return
        paths = list(paths)
        for part in _chunked(paths, 200):
            t.delete("path IN (" + ",".join(_q(p) for p in part) + ")")

    def delete_prefix(self, prefix: str) -> List[str]:
        """Remove every indexed path under a directory. Returns the removed paths."""
        paths = self.manifest_under(prefix)
        self.delete_paths(paths)
        for p in paths:
            self.manifest_delete(p)
        return paths

    def replace_rows(self, paths: Iterable[str], rows: List[dict]) -> None:
        """Atomically-ish swap all rows of `paths` for `rows` (delete, then one add)."""
        self.delete_paths(paths)
        if rows:
            self.table.add(rows)

    def maintain(self) -> None:
        t = self.table
        if t is None:
            return
        try:
            t.optimize()
        except Exception:
            pass
        if t.count_rows() >= cfg.VECTOR_INDEX_MIN_ROWS:
            try:
                has_vec = any(i.index_type != "FTS" for i in t.list_indices())
                if not has_vec:
                    t.create_index(metric="cosine", index_type="IVF_PQ", vector_column_name="vector")
            except Exception:
                pass


@contextmanager
def single_instance(name: str, data_dir: Path = None):
    """Cross-process lock (Windows msvcrt). Yields True if acquired, False if another holds it."""
    import msvcrt
    d = Path(data_dir or cfg.DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    f = open(d / f"{name}.lock", "a+")
    try:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            got = True
        except OSError:
            got = False
        yield got
    finally:
        if got:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        f.close()
