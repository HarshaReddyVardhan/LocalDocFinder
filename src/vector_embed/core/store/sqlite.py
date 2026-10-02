"""SQLite state: manifest, persistent queue, meta, chat sessions, locks, model catalog, usage.

Light enough for the always-on watcher (no ML imports). The schema version lives in
``PRAGMA user_version`` and ``MIGRATIONS`` upgrades older databases at open time.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, Self

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
MAX_STORED_MESSAGES = 400  # per chat session
_OPEN_ATTEMPTS = 8  # opening a database that another process is creating or upgrading
_OPEN_RETRY_SECONDS = 0.05
EMBEDDER_APPROVED_KEY = "embedder_change_approved"  # the model whose index rebuild the user OK'd
PROGRESS_KEY = "last_queue_progress"  # meta key stamped whenever queue items are finished
FAILED_HASH = "failed"  # manifest hash of a file given up on (never equals a real digest)
_FAR_FUTURE = 1e18
STATE_FILENAME = "state.sqlite"
CHAT_LOCK = "chat"  # held while a chat session is active; the indexing worker never runs then

_SCHEMA_V1 = """
CREATE TABLE manifest(
    path TEXT PRIMARY KEY, mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL,
    content_hash TEXT NOT NULL, indexed_at REAL NOT NULL);
CREATE TABLE queue(
    path TEXT PRIMARY KEY, op TEXT NOT NULL, priority REAL NOT NULL DEFAULT 0,
    not_before REAL NOT NULL DEFAULT 0, seq INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE chat_sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at REAL NOT NULL,
    updated_at REAL NOT NULL, context_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE chat_messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL, content TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX chat_messages_session ON chat_messages(session_id, id);
CREATE TABLE locks(
    name TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE models(
    name TEXT NOT NULL, provider TEXT NOT NULL, size_bytes INTEGER, parameter_size TEXT,
    quantization TEXT, context_length INTEGER, discovered_at REAL NOT NULL,
    PRIMARY KEY(provider, name));
CREATE TABLE model_capabilities(
    provider TEXT NOT NULL, name TEXT NOT NULL, capability TEXT NOT NULL,
    PRIMARY KEY(provider, name, capability));
CREATE TABLE usage(
    id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, provider TEXT NOT NULL,
    model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL);
"""

Migration = Callable[[sqlite3.Connection], None]
STALE_PATHS_KEY = "stale_paths"  # case-variant duplicates whose search rows must be removed


def _migrate_case_insensitive_paths(sql: sqlite3.Connection) -> None:
    """Windows paths ignore case: key the manifest and queue that way, and add the claim index.

    Rows that differ only by case collapse into one (the newest wins). The dropped spellings are
    remembered so the worker can remove their search rows, which would otherwise be duplicates.
    """
    sql.execute(
        "CREATE TABLE manifest_new(path TEXT COLLATE NOCASE PRIMARY KEY, mtime_ns INTEGER NOT NULL,"
        " size INTEGER NOT NULL, content_hash TEXT NOT NULL, indexed_at REAL NOT NULL)"
    )
    sql.execute(
        "INSERT OR REPLACE INTO manifest_new SELECT path,mtime_ns,size,content_hash,indexed_at "
        "FROM manifest ORDER BY indexed_at"
    )
    dropped = [
        row[0]
        for row in sql.execute(
            "SELECT m.path FROM manifest m "
            "LEFT JOIN manifest_new n ON n.path = m.path COLLATE BINARY WHERE n.path IS NULL"
        )
    ]
    sql.execute("DROP TABLE manifest")
    sql.execute("ALTER TABLE manifest_new RENAME TO manifest")
    sql.execute(
        "CREATE TABLE queue_new(path TEXT COLLATE NOCASE PRIMARY KEY, op TEXT NOT NULL,"
        " priority REAL NOT NULL DEFAULT 0, not_before REAL NOT NULL DEFAULT 0,"
        " seq INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0)"
    )
    sql.execute("INSERT OR REPLACE INTO queue_new SELECT * FROM queue ORDER BY seq")
    sql.execute("DROP TABLE queue")
    sql.execute("ALTER TABLE queue_new RENAME TO queue")
    sql.execute("CREATE INDEX queue_claim ON queue(attempts, not_before, priority DESC, seq)")
    if dropped:
        sql.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (STALE_PATHS_KEY, json.dumps(dropped)),
        )


# Index i upgrades schema version i+1 -> i+2; version 1 is created by _SCHEMA_V1.
MIGRATIONS: tuple[Migration, ...] = (_migrate_case_insensitive_paths,)
SCHEMA_VERSION = 1 + len(MIGRATIONS)


class QueueItem(NamedTuple):
    path: str
    op: str
    seq: int


class ManifestEntry(NamedTuple):
    mtime_ns: int
    size: int
    content_hash: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    created_at: float


@dataclass(frozen=True)
class ChatSession:
    id: int
    title: str
    updated_at: float


@dataclass(frozen=True)
class UsageTotal:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class StateError(RuntimeError):
    """The state database is unusable (for example, written by a newer version)."""


class StateDb:
    """Thread-safe wrapper around the state database."""

    def __init__(self, data_dir: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._lock = threading.RLock()
        self._sql = sqlite3.connect(
            self.data_dir / STATE_FILENAME,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            self._open_with_retry()
        except BaseException:
            self._sql.close()
            raise

    def _open_with_retry(self) -> None:
        """Switch to WAL and upgrade the schema. Several processes may open a fresh database at
        the same moment; SQLite then answers "locked" at once for some steps (the busy timeout
        does not apply), so retry briefly."""
        for attempt in range(_OPEN_ATTEMPTS):
            try:
                self._sql.execute("PRAGMA journal_mode=WAL")
                self._sql.execute("PRAGMA synchronous=NORMAL")
                self._sql.execute("PRAGMA foreign_keys=ON")
                self._upgrade()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == _OPEN_ATTEMPTS - 1:
                    raise
                time.sleep(_OPEN_RETRY_SECONDS * (attempt + 1))

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One write transaction: ``BEGIN IMMEDIATE`` (take the write lock now, so concurrent
        processes queue up instead of failing half-way), commit on success, roll back on error."""
        with self._lock:
            self._sql.execute("BEGIN IMMEDIATE")
            try:
                yield self._sql
            except BaseException:
                self._sql.execute("ROLLBACK")
                raise
            self._sql.execute("COMMIT")

    def _upgrade(self) -> None:
        # The version is read *inside* the write lock: when the watcher, the worker and the app
        # all open a fresh database at once, exactly one of them creates and upgrades it.
        with self._tx() as sql:
            version = int(sql.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StateError(f"state schema {version} is newer than supported {SCHEMA_VERSION}")
            if version == 0:
                for statement in _SCHEMA_V1.split(";"):
                    if statement.strip():
                        sql.execute(statement)
                version = 1
            for target in range(version + 1, SCHEMA_VERSION + 1):
                MIGRATIONS[target - 2](sql)
            sql.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        with self._lock:
            self._sql.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> tuple[Any, ...] | None:
        with self._lock:
            row: tuple[Any, ...] | None = self._sql.execute(sql, params).fetchone()
        return row

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return self._sql.execute(sql, params).fetchall()

    def _run(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._sql.execute(sql, params)

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self._one("SELECT value FROM meta WHERE key=?", (key,))
        return str(row[0]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        self._run(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def delete_meta(self, key: str) -> None:
        self._run("DELETE FROM meta WHERE key=?", (key,))

    # ------------------------------------------------------------------ manifest
    def manifest_get(self, path: str) -> ManifestEntry | None:
        row = self._one("SELECT mtime_ns,size,content_hash FROM manifest WHERE path=?", (path,))
        return ManifestEntry(*row) if row else None

    def manifest_all(self) -> dict[str, tuple[int, int]]:
        """Every indexed path (keyed in lower case, as Windows compares paths) -> (mtime, size)."""
        rows = self._all("SELECT path,mtime_ns,size FROM manifest")
        return {os.path.normcase(p): (m, s) for p, m, s in rows}

    def canonical_path(self, path: str) -> str:
        """The spelling the index already uses for ``path`` (``path`` itself if it is new).

        Search rows are keyed by the exact string, so one file must always use one spelling.
        """
        row = self._one("SELECT path FROM manifest WHERE path=?", (path,))
        return str(row[0]) if row else path

    def take_stale_paths(self) -> list[str]:
        """Spellings dropped by the case-insensitive migration, once; the caller deletes them."""
        raw = self.get_meta(STALE_PATHS_KEY)
        if raw is None:
            return []
        self._run("DELETE FROM meta WHERE key=?", (STALE_PATHS_KEY,))
        paths: list[str] = json.loads(raw)
        return paths

    def manifest_set(self, path: str, mtime_ns: int, size: int, content_hash: str) -> None:
        self._run(
            "INSERT INTO manifest(path,mtime_ns,size,content_hash,indexed_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(path) DO UPDATE SET mtime_ns=excluded.mtime_ns,size=excluded.size,"
            "content_hash=excluded.content_hash,indexed_at=excluded.indexed_at",
            (path, mtime_ns, size, content_hash, self._clock()),
        )

    def manifest_delete(self, path: str) -> None:
        self._run("DELETE FROM manifest WHERE path=?", (path,))

    def manifest_clear(self) -> None:
        self._run("DELETE FROM manifest")

    def manifest_under(self, prefix: str) -> list[str]:
        """Manifest paths inside a directory (case-insensitive, as Windows paths are)."""
        prefix = prefix.rstrip("\\/") + os.sep
        rows = self._all(
            "SELECT path FROM manifest WHERE substr(path,1,?) = ? COLLATE NOCASE",
            (len(prefix), prefix),
        )
        return [r[0] for r in rows]

    def manifest_count(self) -> int:
        return int(self._one("SELECT COUNT(*) FROM manifest")[0])  # type: ignore[index]

    # ------------------------------------------------------------------ queue
    _UPSERT_QUEUE = (
        "INSERT INTO queue(path,op,priority,not_before,seq,attempts) VALUES(?,?,?,?,?,0) "
        "ON CONFLICT(path) DO UPDATE SET op=excluded.op,priority=excluded.priority,"
        "not_before=excluded.not_before,seq=excluded.seq,attempts=0"
    )

    def enqueue(
        self, path: str, op: str = "upsert", priority: float = 0.0, delay: float = 0.0
    ) -> None:
        """Queue ``path``; re-queueing resets the debounce and attempt count."""
        self._run(self._UPSERT_QUEUE, (path, op, priority, self._clock() + delay, time.time_ns()))

    def enqueue_many(self, items: Iterable[tuple[str, str, float]]) -> int:
        """Queue ``(path, op, priority)`` triples in one transaction; returns how many."""
        count = 0
        now = self._clock()
        with self._tx() as sql:
            for path, op, priority in items:
                sql.execute(self._UPSERT_QUEUE, (path, op, priority, now, time.time_ns()))
                count += 1
        return count

    def queue_size(self, due_only: bool = False) -> int:
        if due_only:  # the same rows ``claim`` would hand out: dead rows are not work
            row = self._one(
                "SELECT COUNT(*) FROM queue WHERE not_before<=? AND attempts<?",
                (self._clock(), _MAX_ATTEMPTS),
            )
        else:
            row = self._one("SELECT COUNT(*) FROM queue")
        return int(row[0]) if row else 0

    def claim(self, limit: int, ignore_debounce: bool = False) -> list[QueueItem]:
        """Highest priority first. Rows stay queued until ``done`` so a crash loses nothing."""
        cutoff = _FAR_FUTURE if ignore_debounce else self._clock()
        rows = self._all(
            "SELECT path,op,seq FROM queue WHERE not_before<=? AND attempts<? "
            "ORDER BY priority DESC, seq LIMIT ?",
            (cutoff, _MAX_ATTEMPTS, limit),
        )
        return [QueueItem(*r) for r in rows]

    def done(self, items: Iterable[tuple[str, int]]) -> None:
        """Remove processed rows, unless a path was re-queued (its ``seq`` changed) meanwhile."""
        finished = list(items)
        with self._tx() as sql:
            for path, seq in finished:
                sql.execute("DELETE FROM queue WHERE path=? AND seq=?", (path, seq))
        if finished:  # lets the watcher tell a worker that worked from one that only exited
            self.set_meta(PROGRESS_KEY, str(time.time_ns()))

    def fail(self, path: str, delay: float = 300.0) -> None:
        """Count a failed attempt and back off; the third strike gives up on this version."""
        self._run(
            "UPDATE queue SET attempts=attempts+1, not_before=? WHERE path=?",
            (self._clock() + delay, path),
        )
        row = self._one("SELECT attempts FROM queue WHERE path=?", (path,))
        if row and row[0] >= _MAX_ATTEMPTS:
            self._give_up(path)

    def defer(self, path: str, delay: float) -> None:
        """Retry later without counting an attempt (the file is fine; the environment is not)."""
        self._run("UPDATE queue SET not_before=? WHERE path=?", (self._clock() + delay, path))

    def _give_up(self, path: str) -> None:
        """Stop retrying a file that keeps failing: record it as seen, until it changes.

        The queue row is removed and the manifest remembers this exact version, so neither the
        watcher nor a reconcile re-queues it. Editing the file makes it eligible again.
        """
        try:
            info = Path(path).stat()
        except OSError:
            self.manifest_delete(path)
        else:
            self.manifest_set(path, info.st_mtime_ns, info.st_size, FAILED_HASH)
        self._run("DELETE FROM queue WHERE path=?", (path,))
        logger.warning("giving up on %s after %d failed attempts", path, _MAX_ATTEMPTS)

    # ------------------------------------------------------------------ locks
    def acquire_lock(self, name: str, owner: str, ttl_seconds: float) -> bool:
        """Take (or refresh, when ``owner`` already holds it) a named lease lock."""
        now = self._clock()
        with self._lock:
            row = self._sql.execute(
                "SELECT owner,expires_at FROM locks WHERE name=?", (name,)
            ).fetchone()
            if row and row[1] > now and row[0] != owner:
                return False
            self._sql.execute(
                "INSERT INTO locks(name,owner,expires_at) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,"
                "expires_at=excluded.expires_at",
                (name, owner, now + ttl_seconds),
            )
        return True

    def release_lock(self, name: str, owner: str) -> None:
        self._run("DELETE FROM locks WHERE name=? AND owner=?", (name, owner))

    def lock_held(self, name: str) -> bool:
        row = self._one("SELECT expires_at FROM locks WHERE name=?", (name,))
        return bool(row and row[0] > self._clock())

    # ------------------------------------------------------------------ chat sessions
    def create_session(self, title: str, context: dict[str, Any] | None = None) -> int:
        now = self._clock()
        with self._lock:
            cursor = self._sql.execute(
                "INSERT INTO chat_sessions(title,created_at,updated_at,context_json) "
                "VALUES(?,?,?,?)",
                (title, now, now, json.dumps(context or {})),
            )
        return int(cursor.lastrowid or 0)

    def add_message(self, session_id: int, role: str, content: str) -> None:
        now = self._clock()
        with self._lock:
            self._sql.execute(
                "INSERT INTO chat_messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                (session_id, role, content, now),
            )
            self._sql.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (now, session_id))
            # A chat that runs for weeks must not grow without limit; the prompt only ever uses
            # the newest messages that fit its token budget anyway.
            self._sql.execute(
                "DELETE FROM chat_messages WHERE session_id=? AND id NOT IN "
                "(SELECT id FROM chat_messages WHERE session_id=? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, MAX_STORED_MESSAGES),
            )

    def messages(self, session_id: int) -> list[ChatMessage]:
        rows = self._all(
            "SELECT role,content,created_at FROM chat_messages WHERE session_id=? ORDER BY id",
            (session_id,),
        )
        return [ChatMessage(*r) for r in rows]

    def session_context(self, session_id: int) -> dict[str, Any]:
        row = self._one("SELECT context_json FROM chat_sessions WHERE id=?", (session_id,))
        return dict(json.loads(row[0])) if row else {}

    def sessions(self, limit: int = 50) -> list[ChatSession]:
        rows = self._all(
            "SELECT id,title,updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [ChatSession(*r) for r in rows]

    # ------------------------------------------------------------------ model catalog
    def replace_models(self, provider: str, models: Iterable[dict[str, Any]]) -> None:
        """Replace the discovered model list of ``provider``.

        Each item: ``name`` plus optional ``size_bytes``, ``parameter_size``, ``quantization``,
        ``context_length`` and ``capabilities`` (list of strings).
        """
        now = self._clock()
        with self._tx() as sql:
            sql.execute("DELETE FROM models WHERE provider=?", (provider,))
            sql.execute("DELETE FROM model_capabilities WHERE provider=?", (provider,))
            for item in models:
                sql.execute(
                    "INSERT INTO models VALUES(?,?,?,?,?,?,?)",
                    (
                        item["name"],
                        provider,
                        item.get("size_bytes"),
                        item.get("parameter_size"),
                        item.get("quantization"),
                        item.get("context_length"),
                        now,
                    ),
                )
                for capability in item.get("capabilities", ()):
                    sql.execute(
                        "INSERT OR IGNORE INTO model_capabilities VALUES(?,?,?)",
                        (provider, item["name"], capability),
                    )

    def list_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        columns = (
            "SELECT name,provider,size_bytes,parameter_size,quantization,context_length FROM models"
        )
        if provider:
            rows = self._all(columns + " WHERE provider=? ORDER BY provider,name", (provider,))
        else:
            rows = self._all(columns + " ORDER BY provider,name")
        out: list[dict[str, Any]] = []
        for name, prov, size, params_, quant, ctx in rows:
            caps = self._all(
                "SELECT capability FROM model_capabilities WHERE provider=? AND name=? "
                "ORDER BY capability",
                (prov, name),
            )
            out.append(
                {
                    "name": name,
                    "provider": prov,
                    "size_bytes": size,
                    "parameter_size": params_,
                    "quantization": quant,
                    "context_length": ctx,
                    "capabilities": [c[0] for c in caps],
                }
            )
        return out

    # ------------------------------------------------------------------ usage
    def record_usage(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float = 0.0,
    ) -> None:
        self._run(
            "INSERT INTO usage(at,provider,model,prompt_tokens,completion_tokens,cost_usd) "
            "VALUES(?,?,?,?,?,?)",
            (self._clock(), provider, model, prompt_tokens, completion_tokens, cost_usd),
        )

    def usage_totals(self, since: float = 0.0) -> list[UsageTotal]:
        rows = self._all(
            "SELECT provider,model,SUM(prompt_tokens),SUM(completion_tokens),SUM(cost_usd) "
            "FROM usage WHERE at>=? GROUP BY provider,model ORDER BY provider,model",
            (since,),
        )
        return [UsageTotal(p, m, int(a), int(b), float(c)) for p, m, a, b, c in rows]

    def spend_since(self, since: float, provider: str | None = None) -> float:
        if provider:
            row = self._one(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage WHERE at>=? AND provider=?",
                (since, provider),
            )
        else:
            row = self._one("SELECT COALESCE(SUM(cost_usd),0) FROM usage WHERE at>=?", (since,))
        return float(row[0]) if row else 0.0
