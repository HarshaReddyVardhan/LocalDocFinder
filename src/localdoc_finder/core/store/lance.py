"""LanceDB tables: ``chunks`` (vectors, text, FTS) and ``documents`` (one row per file).

Vectors from different embedding models cannot be mixed, so the model id and dimension are
recorded in the state DB; a change wipes both tables and the manifest (a clean re-index).
"""

import logging
import shutil
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pyarrow as pa

from localdoc_finder.core.model_names import same_model
from localdoc_finder.core.store.search_columns import SQL_COLUMNS
from localdoc_finder.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

CHUNKS = "chunks"
DOCUMENTS = "documents"
CHUNK_VECTOR = "vector"
DOC_VECTOR = "doc_vector"
LANCE_DIRNAME = "lance"

CHUNK_COLUMNS = [
    "text",
    "path",
    "project",
    "kind",
    "source",
    "ext",
    "symbol",
    "start_line",
    "end_line",
    "page",
    "chunk_hash",
    "model_id",
    "mtime",
    "content_chars",
    "name",
    "search_text",
]
DOCUMENT_COLUMNS = [
    "path",
    "project",
    "doc_type",
    "title",
    "full_text",
    "version_group",
    "modified_at",
    "model_id",
]

_KEEP_VERSIONS = timedelta(hours=1)  # older table versions are deleted at optimize
_REINDEX_GROWTH = 2  # rebuild the vector index when the table has grown this many times
_BATCH = 200
_READ_ATTEMPTS = 3
_READ_RETRY_SECONDS = 0.3
_VECTOR_INDEX_MIN_ROWS = 100_000
_T = TypeVar("_T")

Row = dict[str, Any]
LanceDb: TypeAlias = Any  # a lancedb connection
LanceTable: TypeAlias = Any  # lancedb ships no usable type information


def sql_quote(value: str) -> str:
    """Quote a string literal for a LanceDB SQL filter."""
    return "'" + value.replace("'", "''") + "'"


def _batches(items: Sequence[_T], size: int = _BATCH) -> Iterator[Sequence[_T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def chunk_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            (CHUNK_VECTOR, pa.list_(pa.float32(), dim)),
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
            ("content_chars", pa.int32()),
            ("name", pa.string()),
            ("search_text", pa.string()),
        ]
    )


def document_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            (DOC_VECTOR, pa.list_(pa.float32(), dim)),
            ("path", pa.string()),
            ("project", pa.string()),
            ("doc_type", pa.string()),
            ("title", pa.string()),
            ("full_text", pa.string()),
            ("version_group", pa.string()),
            ("modified_at", pa.int64()),
            ("model_id", pa.string()),
        ]
    )


class ModelMismatchError(RuntimeError):
    """The index was built with another embedding model or dimension than the one configured."""


class IndexRebuildingError(RuntimeError):
    """The configured embedder differs from the one that built the index; until the rebuild
    replaces the old vectors, a query vector cannot be compared with them."""


class IndexSchemaError(RuntimeError):
    """The index was written by a newer version of the app than this one."""


def _add_search_columns(store: "LanceStore") -> None:
    """v1 -> v2: ``content_chars``, ``name`` and ``search_text`` on chunks, backfilled from the
    stored text and path (no re-embedding), and BM25 moved from ``text`` to ``search_text``."""
    table = store.table(CHUNKS)
    if table is None:
        return
    present = set(table.schema.names)
    missing = {column: sql for column, sql in SQL_COLUMNS.items() if column not in present}
    if missing:  # a resumed upgrade skips what it already added
        table.add_columns(missing)
    store.create_fts(CHUNKS)
    for index in table.list_indices():
        if list(getattr(index, "columns", None) or ()) == ["text"]:
            table.drop_index(index.name)


LanceMigration = Callable[["LanceStore"], None]
# Index i upgrades the index schema from version i+1 to i+2. Version 1 is the first layout of the
# chunk and document tables; a change to ``chunk_schema``/``document_schema`` adds a step here.
LANCE_MIGRATIONS: tuple[LanceMigration, ...] = (_add_search_columns,)
LANCE_SCHEMA_VERSION = 1 + len(LANCE_MIGRATIONS)
SCHEMA_VERSION_KEY = "lance_schema_version"


def reset_index(data_dir: Path, state: StateDb, db: LanceDb | None = None) -> None:
    """Delete the whole index and queue every file again (the user's "rebuild from scratch").

    Only the vectors and the list of indexed files go; settings, chat history, pinned items and
    the user's own files are untouched. The worker must not be running.
    """
    import lancedb

    root = Path(data_dir) / LANCE_DIRNAME
    connection = db if db is not None else lancedb.connect(str(root))
    try:
        listing = (
            connection.list_tables()
            if hasattr(connection, "list_tables")
            else connection.table_names()
        )
        names = list(getattr(listing, "tables", listing))
    except Exception:  # a damaged folder: fall through to deleting it
        logger.warning("lance: cannot list tables while resetting", exc_info=True)
        names = []
    for name in names:
        try:
            connection.drop_table(name)
        except Exception:  # a damaged table may not drop cleanly: remove its files instead
            logger.warning("lance: drop_table(%s) failed; deleting its folder", name, exc_info=True)
    for leftover in root.glob("*.lance"):  # anything the drop could not remove
        shutil.rmtree(leftover, ignore_errors=True)
    state.manifest_clear()
    state.clear_queue()
    state.set_meta("last_reconcile", "0")  # the next run scans every folder again
    state.delete_meta(SCHEMA_VERSION_KEY)
    for name in (CHUNKS, DOCUMENTS):
        state.delete_meta(f"vector_index_rows_{name}")


class LanceStore:
    """Chunk and document tables bound to one embedding model.

    Pass ``dim=None`` for read-only use (search): existing tables are opened as they are.
    """

    def __init__(
        self,
        data_dir: Path,
        state: StateDb,
        model_id: str,
        dim: int | None = None,
        vector_index_min_rows: int = _VECTOR_INDEX_MIN_ROWS,
        *,
        allow_wipe: bool = False,
    ) -> None:
        import lancedb

        self.model_id = model_id
        self.dim = dim
        self._allow_wipe = allow_wipe
        self._state = state
        self._min_rows = vector_index_min_rows
        self._root = Path(data_dir) / LANCE_DIRNAME
        self.db: Any = lancedb.connect(str(self._root))
        self._tables: dict[str, LanceTable] = {}
        if dim is None:
            self._open_existing()
        else:
            self.check_model()

    # ------------------------------------------------------------------ tables
    def _table_names(self) -> list[str]:
        listing = (
            self.db.list_tables() if hasattr(self.db, "list_tables") else self.db.table_names()
        )
        return list(getattr(listing, "tables", listing))

    def _open_existing(self) -> None:
        names = self._table_names()
        if names:
            self._stored_schema_version()  # refuses an index from a newer app; never migrates
        for name in (CHUNKS, DOCUMENTS):
            if name in names:
                self._tables[name] = self.db.open_table(name)

    def _stored_schema_version(self) -> int:
        """The index's schema version (1 for an index from before versions were recorded)."""
        stored = int(self._state.get_meta(SCHEMA_VERSION_KEY, "1") or 1)
        if stored > LANCE_SCHEMA_VERSION:
            raise IndexSchemaError(
                f"the index has schema version {stored}, newer than this app supports "
                f"({LANCE_SCHEMA_VERSION}); update the app"
            )
        return stored

    def _migrate(self, migrations: Sequence[LanceMigration] | None = None) -> None:
        """Bring existing tables up to ``LANCE_SCHEMA_VERSION``, one recorded step at a time, so
        an interrupted upgrade resumes where it stopped."""
        steps = LANCE_MIGRATIONS if migrations is None else migrations
        latest = 1 + len(steps)
        version = self._stored_schema_version()
        while version < latest:
            logger.info("lance: migrating the index schema %d -> %d", version, version + 1)
            steps[version - 1](self)
            version += 1
            self._state.set_meta(SCHEMA_VERSION_KEY, str(version))

    def check_model(self) -> bool:
        """Create tables; rebuild the index if the model or dimension changed (True if wiped).

        A different model or dimension makes every stored vector meaningless, but wiping is
        destructive and slow to undo, so it only happens when the user approved it (the
        ``allow_wipe`` flag). Otherwise this raises and leaves the index as it is.
        """
        assert self.dim is not None
        stored = (self._state.get_meta("model_id"), self._state.get_meta("dim"))
        wanted = (self.model_id, str(self.dim))
        names = self._table_names()
        untracked = stored == (None, None)  # tables with no record of their model: nothing to keep
        wiped = bool(names) and stored != wanted
        if wiped and not (self._allow_wipe or untracked):
            raise ModelMismatchError(
                f"the index was built with {stored[0]} (dim {stored[1]}) but {wanted[0]} "
                f"(dim {wanted[1]}) is configured; switch with: ldf models --embedder "
                f"{wanted[0]} --yes (this re-indexes every file)"
            )
        if not wiped and names and not self._tables_open_cleanly(names):
            wiped = True  # a crash left a table unreadable; the index is derived data, so rebuild
        if wiped:
            self.wipe()
            names = []
        schemas = {CHUNKS: chunk_schema(self.dim), DOCUMENTS: document_schema(self.dim)}
        for name, schema in schemas.items():
            if name in names:
                self._tables[name] = self.db.open_table(name)
            else:
                self._tables[name] = self.db.create_table(name, schema=schema)
                self.create_fts(name)
        if names:
            self._migrate()
        else:  # new tables are created in the current layout
            self._state.set_meta(SCHEMA_VERSION_KEY, str(LANCE_SCHEMA_VERSION))
        self._state.set_meta("model_id", self.model_id)
        self._state.set_meta("dim", str(self.dim))
        return wiped

    def _tables_open_cleanly(self, names: Sequence[str]) -> bool:
        """Whether every table can be opened and read; a half-written one (power loss, a killed
        process) is reported rather than left to crash the first search."""
        for name in names:
            try:
                self.db.open_table(name).search().limit(1).to_list()  # reads real data
            except Exception:  # lancedb raises RuntimeError/OSError/ValueError depending on damage
                logger.error(
                    "lance: table %s is unreadable; rebuilding the index", name, exc_info=True
                )
                return False
        return True

    def wipe(self) -> None:
        """Drop both tables and forget what was indexed, so every file is indexed again."""
        reset_index(self._root.parent, self._state, self.db)
        self._tables.clear()

    def _fts_column(self, name: str) -> str:
        """The column BM25 searches: ``search_text`` on chunks (``text`` on an index from before
        it existed, until the upgrade has run), ``full_text`` on documents."""
        if name != CHUNKS:
            return "full_text"
        table = self._tables.get(name)
        if table is not None and "search_text" not in table.schema.names:
            return "text"
        return "search_text"

    def create_fts(self, name: str) -> None:
        """BM25 index; an optimisation only, so failure degrades to vector-only search.

        Chunk text is stemmed with English stop words removed, so "payments" finds "payment";
        whole documents keep exact words (they are matched by name and phrase, not ranked).
        """
        from lancedb.index import FTS

        column = self._fts_column(name)
        stemmed = name == CHUNKS
        base: dict[str, Any] = {
            "with_position": True,
            "language": "English",
            "stem": stemmed,
            "remove_stop_words": stemmed,
        }
        attempts: list[dict[str, Any]] = [{**base, "split_identifiers": True}, base, {}]
        for options in attempts:
            try:
                self._tables[name].create_index(column, config=FTS(**options), replace=True)
            except TypeError:
                continue  # older lancedb without this option
            except Exception:
                logger.warning("lance: FTS index on %s.%s failed", name, column, exc_info=True)
                return
            else:
                return

    def vectors_current(self) -> bool:
        """Whether the stored vectors come from the configured embedder.

        Read on every call: a long-lived search front-end must notice the worker finishing (or
        starting) a rebuild. An index with no recorded model has nothing to mix up.
        """
        stored = self._state.get_meta("model_id")
        return not stored or same_model(stored, self.model_id)

    def require_current_vectors(self) -> None:
        if not self.vectors_current():
            raise IndexRebuildingError(
                f"the index is being rebuilt for {self.model_id}; search works again once "
                "the first indexing pass has finished"
            )

    def table(self, name: str) -> LanceTable | None:
        """The named table, or ``None`` if the index has not been built yet."""
        if name not in self._tables:
            self._open_existing()
        return self._tables.get(name)

    @property
    def chunks(self) -> LanceTable | None:
        return self.table(CHUNKS)

    @property
    def documents(self) -> LanceTable | None:
        return self.table(DOCUMENTS)

    def count(self, name: str = CHUNKS) -> int:
        return self._read(name, lambda table: int(table.count_rows()), 0)

    def _read(self, name: str, action: Callable[[Any], _T], default: _T) -> _T:
        """Run a read on a table, re-opening it and retrying when it changed underneath us.

        The worker (another process) compacts, cleans old versions and may rebuild the index
        while the app searches; a handle opened before that points at files that are gone. The
        retry opens a fresh handle; a persistent failure returns ``default`` instead of crashing
        the caller.
        """
        for attempt in range(_READ_ATTEMPTS):
            table = self.table(name)
            if table is None:
                return default
            try:
                return action(table)
            except Exception:
                self._tables.pop(name, None)  # the cached handle is stale
                if attempt == _READ_ATTEMPTS - 1:
                    logger.warning("lance: reading %s failed", name, exc_info=True)
                    return default
                time.sleep(_READ_RETRY_SECONDS * (attempt + 1))
        return default

    # ------------------------------------------------------------------ chunks
    def vectors_for_hashes(self, hashes: Iterable[str]) -> dict[str, np.ndarray]:
        """Embeddings already stored for identical chunk text (embedded once, reused anywhere)."""
        table = self.chunks
        found: dict[str, np.ndarray] = {}
        if table is None:
            return found
        for part in _batches(sorted(set(hashes))):
            listed = ",".join(sql_quote(h) for h in part)
            where = f"chunk_hash IN ({listed}) AND model_id = {sql_quote(self.model_id)}"
            limit = len(part) * 4
            rows: list[Row] = self._read(
                CHUNKS, partial(self._rows_by_where, where=where, limit=limit), []
            )
            for row in rows:
                found.setdefault(row["chunk_hash"], np.asarray(row[CHUNK_VECTOR], dtype=np.float32))
        return found

    @staticmethod
    def _rows_by_where(table: LanceTable, *, where: str, limit: int) -> list[Row]:
        rows: list[Row] = (
            table.search().where(where).select(["chunk_hash", CHUNK_VECTOR]).limit(limit).to_list()
        )
        return rows

    def delete_paths(self, paths: Iterable[str]) -> None:
        """Remove every chunk and document row belonging to ``paths``."""
        listed = list(paths)
        for name in (CHUNKS, DOCUMENTS):
            table = self.table(name)
            if table is None:
                continue
            for part in _batches(listed):
                table.delete("path IN (" + ",".join(sql_quote(p) for p in part) + ")")

    def doc_types_for(self, paths: Sequence[str]) -> dict[str, str]:
        """The stored document type of each indexed path (paths never indexed are absent)."""
        found: dict[str, str] = {}
        for part in _batches(list(paths)):
            listed = ",".join(sql_quote(p) for p in part)
            for row in self.scan(DOCUMENTS, ["path", "doc_type"], f"path IN ({listed})", len(part)):
                found[row["path"]] = row["doc_type"]
        return found

    def replace_rows(self, paths: Iterable[str], rows: list[Row]) -> None:
        """Swap all chunk rows of ``paths`` for ``rows`` (delete, then a single add)."""
        table = self.chunks
        assert table is not None, "store opened read-only"
        for part in _batches(list(paths)):
            table.delete("path IN (" + ",".join(sql_quote(p) for p in part) + ")")
        if rows:
            table.add(rows)

    def replace_documents(self, paths: Iterable[str], rows: list[Row]) -> None:
        """Swap the document rows of ``paths`` for ``rows``."""
        table = self.documents
        assert table is not None, "store opened read-only"
        for part in _batches(list(paths)):
            table.delete("path IN (" + ",".join(sql_quote(p) for p in part) + ")")
        if rows:
            table.add(rows)

    def set_version_groups(self, groups: dict[str, str], candidates: Iterable[str]) -> None:
        """Write ``version_group`` for ``groups``; other ``candidates`` are reset to ungrouped.

        One update per group (not per file): a table update rewrites a fragment each time.
        """
        table = self.documents
        assert table is not None, "store opened read-only"
        by_group: dict[str, list[str]] = {}
        for path in candidates:
            by_group.setdefault(groups.get(path, ""), []).append(path)
        for group, paths in by_group.items():
            for part in _batches(paths):
                listed = ",".join(sql_quote(p) for p in part)
                table.update(where=f"path IN ({listed})", values={"version_group": group})

    # ------------------------------------------------------------------ queries
    def scan(self, name: str, columns: list[str], where: str = "", limit: int = 1000) -> list[Row]:
        def run(table: LanceTable) -> list[Row]:
            query = table.search()
            if where:
                query = query.where(where)
            rows: list[Row] = query.select(columns).limit(limit).to_list()
            return rows

        return self._read(name, run, [])

    def has_column(self, name: str, column: str) -> bool:
        table = self.table(name)
        return table is not None and column in table.schema.names

    def vector_search(
        self,
        name: str,
        vector: np.ndarray,
        columns: list[str],
        where: str = "",
        limit: int = 50,
        *,
        min_content_chars: int = 0,
    ) -> list[Row]:
        """Nearest rows, each with its cosine ``_distance``; ``[]`` while the stored vectors
        belong to another embedder (comparing across models would rank at random).

        ``min_content_chars`` leaves out chunks with less readable text than that (ignored on an
        index from before the column existed).
        """
        if self.table(name) is None:
            return []
        if not self.vectors_current():
            logger.info("lance: vectors are from another embedder; skipping the vector search")
            return []
        if min_content_chars and self.has_column(name, "content_chars"):
            floor = f"content_chars >= {int(min_content_chars)}"
            where = f"({where}) AND {floor}" if where else floor
        column = CHUNK_VECTOR if name == CHUNKS else DOC_VECTOR

        def run(handle: LanceTable) -> list[Row]:
            query = handle.search(vector, vector_column_name=column).metric("cosine")
            if where:
                query = query.where(where, prefilter=True)
            rows: list[Row] = query.select([*columns, "_distance"]).limit(limit).to_list()
            return rows

        return self._read(name, run, [])

    def fts_search(
        self, name: str, text: str, columns: list[str], where: str = "", limit: int = 50
    ) -> list[Row]:
        """BM25 keyword search, each row with its ``_score``; ``[]`` if the FTS index is
        unavailable."""
        fts_column = self._fts_column(name)

        def run(table: LanceTable) -> list[Row]:
            query = table.search(text, query_type="fts", fts_columns=fts_column)
            if where:
                query = query.where(where, prefilter=True)
            rows: list[Row] = query.select([*columns, "_score"]).limit(limit).to_list()
            return rows

        return self._read(name, run, [])

    # ------------------------------------------------------------------ maintenance
    @staticmethod
    def _has_index_on(table: LanceTable, column: str) -> bool:
        """Whether any index covers ``column``. Matching on the column, not the index type: the
        full-text index reports itself as ``INVERTED`` or ``FTS`` depending on the version."""
        return any(column in (getattr(i, "columns", None) or ()) for i in table.list_indices())

    def maintain(self) -> None:
        """Keep the tables fast: keyword and lookup indexes, compaction, the vector index."""
        steps: tuple[tuple[str, Callable[[LanceTable, str], object]], ...] = (
            ("FTS check", self._repair_fts),
            ("scalar indexes", self._scalar_indexes),
            ("optimize", self._optimize),
        )
        for name in (CHUNKS, DOCUMENTS):
            table = self.table(name)
            if table is None:
                continue
            for what, step in steps:
                self._guarded(name, what, partial(step, table, name))
            if table.count_rows() >= self._min_rows:
                self._guarded(name, "vector index", partial(self._vector_index, table, name))

    @staticmethod
    def _guarded(name: str, what: str, action: Callable[[], object]) -> None:
        """Maintenance is an optimisation: a failure is logged and the next run tries again."""
        try:
            action()
        except Exception:
            logger.warning("lance: %s failed on %s", what, name, exc_info=True)

    def _repair_fts(self, table: LanceTable, name: str) -> None:
        if not self._has_index_on(table, self._fts_column(name)):  # its creation failed earlier
            self.create_fts(name)

    @staticmethod
    def _optimize(table: LanceTable, _name: str) -> None:
        table.optimize(cleanup_older_than=_KEEP_VERSIONS)

    def _scalar_indexes(self, table: LanceTable, name: str) -> None:
        """BTREE indexes on the columns used by ``IN (...)`` lookups and deletes."""
        from lancedb.index import BTree

        columns = ("chunk_hash", "path") if name == CHUNKS else ("path",)
        for column in columns:
            if not self._has_index_on(table, column):
                table.create_index(column, config=BTree(), replace=False)

    def _vector_index(self, table: LanceTable, name: str) -> None:
        """Build the vector index, and rebuild it when the table has doubled since it was made
        (new rows are searched by brute force until then; a stale index degrades recall)."""
        from lancedb.index import IvfPq

        column = CHUNK_VECTOR if name == CHUNKS else DOC_VECTOR
        key = f"vector_index_rows_{name}"
        rows = int(table.count_rows())
        built_at = int(self._state.get_meta(key, "0") or 0)
        if self._has_index_on(table, column) and rows < built_at * _REINDEX_GROWTH:
            return
        table.create_index(column, config=IvfPq(distance_type="cosine"), replace=True)
        self._state.set_meta(key, str(rows))
