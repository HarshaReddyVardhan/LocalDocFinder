"""LanceDB tables: ``chunks`` (vectors, text, FTS) and ``documents`` (one row per file).

Vectors from different embedding models cannot be mixed, so the model id and dimension are
recorded in the state DB; a change wipes both tables and the manifest (a clean re-index).
"""

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

import numpy as np
import pyarrow as pa

from vector_embed.core.store.sqlite import StateDb

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

_BATCH = 200
_VECTOR_INDEX_MIN_ROWS = 100_000
_T = TypeVar("_T")

Row = dict[str, Any]
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
    ) -> None:
        import lancedb

        self.model_id = model_id
        self.dim = dim
        self._state = state
        self._min_rows = vector_index_min_rows
        self.db: Any = lancedb.connect(str(Path(data_dir) / LANCE_DIRNAME))
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
        for name in (CHUNKS, DOCUMENTS):
            if name in names:
                self._tables[name] = self.db.open_table(name)

    def check_model(self) -> bool:
        """Create tables, or wipe everything if the model or dimension changed (True if wiped)."""
        assert self.dim is not None
        stored = (self._state.get_meta("model_id"), self._state.get_meta("dim"))
        wanted = (self.model_id, str(self.dim))
        names = self._table_names()
        wiped = bool(names) and stored != wanted
        if wiped:
            for name in names:
                self.db.drop_table(name)
            self._state.manifest_clear()
            names = []
        schemas = {CHUNKS: chunk_schema(self.dim), DOCUMENTS: document_schema(self.dim)}
        for name, schema in schemas.items():
            if name in names:
                self._tables[name] = self.db.open_table(name)
            else:
                self._tables[name] = self.db.create_table(name, schema=schema)
                self._create_fts(name, "text" if name == CHUNKS else "full_text")
        self._state.set_meta("model_id", self.model_id)
        self._state.set_meta("dim", str(self.dim))
        return wiped

    def _create_fts(self, name: str, column: str) -> None:
        """BM25 index; an optimisation only, so failure degrades to vector-only search."""
        from lancedb.index import FTS

        attempts: list[dict[str, Any]] = [
            {"stem": False, "remove_stop_words": False, "with_position": True,
             "split_identifiers": True},
            {"stem": False, "remove_stop_words": False, "with_position": True},
            {},
        ]  # fmt: skip
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
        table = self.table(name)
        return int(table.count_rows()) if table is not None else 0

    # ------------------------------------------------------------------ chunks
    def existing_hashes(self, path: str) -> set[str]:
        table = self.chunks
        if table is None:
            return set()
        rows = (
            table.search()
            .where(f"path = {sql_quote(path)}")
            .select(["chunk_hash"])
            .limit(100_000)
            .to_list()
        )
        return {r["chunk_hash"] for r in rows}

    def vectors_for_hashes(self, hashes: Iterable[str]) -> dict[str, np.ndarray]:
        """Embeddings already stored for identical chunk text (embedded once, reused anywhere)."""
        table = self.chunks
        found: dict[str, np.ndarray] = {}
        if table is None:
            return found
        for part in _batches(sorted(set(hashes))):
            listed = ",".join(sql_quote(h) for h in part)
            where = f"chunk_hash IN ({listed}) AND model_id = {sql_quote(self.model_id)}"
            rows = (
                table.search()
                .where(where)
                .select(["chunk_hash", CHUNK_VECTOR])
                .limit(len(part) * 4)
                .to_list()
            )
            for row in rows:
                found.setdefault(row["chunk_hash"], np.asarray(row[CHUNK_VECTOR], dtype=np.float32))
        return found

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

    def delete_prefix(self, prefix: str) -> list[str]:
        """Remove every indexed path under a directory; returns the removed paths."""
        paths = self._state.manifest_under(prefix)
        self.delete_paths(paths)
        for path in paths:
            self._state.manifest_delete(path)
        return paths

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
        """Write ``version_group`` for ``groups``; other ``candidates`` are reset to ungrouped."""
        table = self.documents
        assert table is not None, "store opened read-only"
        for path in candidates:
            group = groups.get(path, "")
            table.update(where=f"path = {sql_quote(path)}", values={"version_group": group})

    # ------------------------------------------------------------------ queries
    def scan(self, name: str, columns: list[str], where: str = "", limit: int = 1000) -> list[Row]:
        table = self.table(name)
        if table is None:
            return []
        query = table.search()
        if where:
            query = query.where(where)
        rows: list[Row] = query.select(columns).limit(limit).to_list()
        return rows

    def vector_search(
        self, name: str, vector: np.ndarray, columns: list[str], where: str = "", limit: int = 50
    ) -> list[Row]:
        table = self.table(name)
        if table is None:
            return []
        column = CHUNK_VECTOR if name == CHUNKS else DOC_VECTOR
        query = table.search(vector, vector_column_name=column).metric("cosine")
        if where:
            query = query.where(where, prefilter=True)
        rows: list[Row] = query.select(columns).limit(limit).to_list()
        return rows

    def fts_search(
        self, name: str, text: str, columns: list[str], where: str = "", limit: int = 50
    ) -> list[Row]:
        """BM25 keyword search; returns ``[]`` if the FTS index is unavailable."""
        table = self.table(name)
        if table is None:
            return []
        try:
            query = table.search(text, query_type="fts")
            if where:
                query = query.where(where, prefilter=True)
            rows: list[Row] = query.select(columns).limit(limit).to_list()
        except Exception:
            logger.debug("lance: FTS search failed on %s", name, exc_info=True)
            return []
        return rows

    # ------------------------------------------------------------------ maintenance
    def maintain(self) -> None:
        """Compact files; build an IVF_PQ index once the table is large enough."""
        for name in (CHUNKS, DOCUMENTS):
            table = self.table(name)
            if table is None:
                continue
            try:
                table.optimize()
            except Exception:
                logger.warning("lance: optimize failed on %s", name, exc_info=True)
            if table.count_rows() < self._min_rows:
                continue
            from lancedb.index import IvfPq

            try:
                if not any(i.index_type != "FTS" for i in table.list_indices()):
                    table.create_index(
                        CHUNK_VECTOR if name == CHUNKS else DOC_VECTOR,
                        config=IvfPq(distance_type="cosine"),
                    )
            except Exception:
                logger.warning("lance: vector index failed on %s", name, exc_info=True)
