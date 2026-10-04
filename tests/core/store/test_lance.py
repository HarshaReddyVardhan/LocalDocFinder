from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from localdoc_finder.core.store import lance as lc
from localdoc_finder.core.store.lance import LanceStore, sql_quote
from localdoc_finder.core.store.search_columns import content_chars, file_name, search_text
from localdoc_finder.core.store.sqlite import StateDb

DIM = 4


def chunk(path: str, text: str, vec: list[float], *, h: str | None = None, **kw: Any) -> dict:
    return {
        "vector": np.asarray(vec, dtype=np.float32),
        "text": text,
        "path": path,
        "project": kw.get("project", "p"),
        "kind": kw.get("kind", "code"),
        "source": "",
        "ext": ".py",
        "symbol": "",
        "start_line": 1,
        "end_line": 2,
        "page": 0,
        "chunk_hash": h or f"h-{text}",
        "model_id": "m1",
        "mtime": kw.get("mtime", 1),
        "content_chars": content_chars(text),
        "name": file_name(path),
        "search_text": search_text(path, text),
    }


def document(path: str, text: str, vec: list[float], doc_type: str = "resume") -> dict:
    return {
        "doc_vector": np.asarray(vec, dtype=np.float32),
        "path": path,
        "project": "p",
        "doc_type": doc_type,
        "title": path,
        "full_text": text,
        "version_group": "",
        "modified_at": 1,
        "model_id": "m1",
    }


@pytest.fixture
def state(tmp_path: Path) -> Iterator[StateDb]:
    with StateDb(tmp_path) as s:
        yield s


@pytest.fixture
def store(tmp_path: Path, state: StateDb) -> LanceStore:
    return LanceStore(tmp_path, state, "m1", dim=DIM)


def test_sql_quote_escapes() -> None:
    assert sql_quote("a'b") == "'a''b'"


def test_creates_both_tables_and_records_model(store: LanceStore, state: StateDb) -> None:
    assert store.count() == 0
    assert store.count(lc.DOCUMENTS) == 0
    assert state.get_meta("model_id") == "m1"
    assert state.get_meta("dim") == str(DIM)


def test_replace_and_count(store: LanceStore) -> None:
    store.replace_rows(
        ["a.py"], [chunk("a.py", "one", [1, 0, 0, 0]), chunk("a.py", "two", [0, 1, 0, 0])]
    )
    assert store.count() == 2
    store.replace_rows(["a.py"], [chunk("a.py", "three", [0, 0, 1, 0])])
    assert store.count() == 1
    assert {r["chunk_hash"] for r in store.scan(lc.CHUNKS, ["chunk_hash"])} == {"h-three"}


def test_vectors_are_reused_by_hash(store: LanceStore) -> None:
    store.replace_rows(["a.py"], [chunk("a.py", "one", [1, 0, 0, 0], h="H1")])
    found = store.vectors_for_hashes(["H1", "missing"])
    assert set(found) == {"H1"}
    assert found["H1"].tolist() == [1, 0, 0, 0]


def test_vector_search_ranks_nearest_first(store: LanceStore) -> None:
    store.replace_rows(
        ["a.py", "b.py"],
        [chunk("a.py", "alpha", [1, 0, 0, 0]), chunk("b.py", "beta", [0, 1, 0, 0])],
    )
    rows = store.vector_search(
        lc.CHUNKS, np.asarray([0.9, 0.1, 0, 0], np.float32), ["path"], limit=2
    )
    assert [r["path"] for r in rows] == ["a.py", "b.py"]
    filtered = store.vector_search(
        lc.CHUNKS, np.asarray([1, 0, 0, 0], np.float32), ["path"], where="path = 'b.py'"
    )
    assert [r["path"] for r in filtered] == ["b.py"]


def test_fts_search_finds_exact_identifier(store: LanceStore) -> None:
    store.replace_rows(
        ["a.py", "b.py"],
        [
            chunk("a.py", "def charge_card(amount): pass", [1, 0, 0, 0]),
            chunk("b.py", "def other(): pass", [0, 1, 0, 0]),
        ],
    )
    store.maintain()  # makes new rows visible to the FTS index
    rows = store.fts_search(lc.CHUNKS, "charge_card", ["path"])
    assert [r["path"] for r in rows] == ["a.py"]


def test_scan_with_filter(store: LanceStore) -> None:
    store.replace_rows(
        ["a.py", "b.py"],
        [chunk("a.py", "x", [1, 0, 0, 0], kind="doc"), chunk("b.py", "y", [0, 1, 0, 0])],
    )
    assert [r["path"] for r in store.scan(lc.CHUNKS, ["path"], "kind = 'doc'")] == ["a.py"]
    assert len(store.scan(lc.CHUNKS, ["path"])) == 2


def test_documents_roundtrip_and_delete_paths(store: LanceStore) -> None:
    store.replace_documents(["r.pdf"], [document("r.pdf", "python developer", [1, 0, 0, 0])])
    store.replace_rows(["r.pdf"], [chunk("r.pdf", "python developer", [1, 0, 0, 0])])
    rows = store.vector_search(
        lc.DOCUMENTS,
        np.asarray([1, 0, 0, 0], np.float32),
        ["path", "doc_type"],
        "doc_type = 'resume'",
    )
    assert rows[0]["path"] == "r.pdf"
    store.delete_paths(["r.pdf"])
    assert store.count() == 0
    assert store.count(lc.DOCUMENTS) == 0


def test_a_model_change_without_approval_refuses_and_keeps_the_index(
    tmp_path: Path, state: StateDb
) -> None:
    first = LanceStore(tmp_path, state, "m1", dim=DIM)
    first.replace_rows(["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])])
    state.manifest_set("a.py", 1, 1, "h")
    with pytest.raises(lc.ModelMismatchError, match="m2"):
        LanceStore(tmp_path, state, "m2", dim=DIM)
    assert LanceStore(tmp_path, state, "m1", dim=DIM).count() == 1  # nothing was destroyed
    assert state.manifest_count() == 1
    assert state.get_meta("model_id") == "m1"


def test_a_dimension_change_is_refused_too(tmp_path: Path, state: StateDb) -> None:
    LanceStore(tmp_path, state, "m1", dim=DIM).replace_rows(
        ["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])]
    )
    with pytest.raises(lc.ModelMismatchError, match="dim"):
        LanceStore(tmp_path, state, "m1", dim=DIM + 4)


def test_an_approved_model_change_wipes_and_requeues_everything(
    tmp_path: Path, state: StateDb
) -> None:
    first = LanceStore(tmp_path, state, "m1", dim=DIM)
    first.replace_rows(["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])])
    state.manifest_set("a.py", 1, 1, "h")
    state.set_meta("last_reconcile", "12345")
    second = LanceStore(tmp_path, state, "m2", dim=DIM, allow_wipe=True)
    assert second.count() == 0
    assert state.manifest_count() == 0
    assert state.get_meta("model_id") == "m2"
    assert state.get_meta("last_reconcile") == "0"  # a reconcile is due: every file comes back


def test_tables_without_any_record_of_their_model_are_adopted_not_blocked(
    tmp_path: Path, state: StateDb
) -> None:
    LanceStore(tmp_path, state, "m1", dim=DIM)
    state.delete_meta("model_id")
    state.delete_meta("dim")
    assert LanceStore(tmp_path, state, "m9", dim=DIM).count() == 0  # no trustworthy vectors to keep


def test_reopen_same_model_keeps_data(tmp_path: Path, state: StateDb) -> None:
    LanceStore(tmp_path, state, "m1", dim=DIM).replace_rows(
        ["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])]
    )
    again = LanceStore(tmp_path, state, "m1", dim=DIM)
    assert again.count() == 1


def test_read_only_open_without_index_returns_empty(tmp_path: Path, state: StateDb) -> None:
    ro = LanceStore(tmp_path, state, "m1", dim=None)
    assert ro.chunks is None
    assert ro.count() == 0
    assert ro.vectors_for_hashes(["x"]) == {}
    assert ro.scan(lc.CHUNKS, ["path"]) == []
    assert ro.vector_search(lc.CHUNKS, np.zeros(DIM, np.float32), ["path"]) == []
    assert ro.fts_search(lc.CHUNKS, "x", ["path"]) == []
    ro.delete_paths(["a"])
    ro.maintain()


def test_read_only_sees_tables_built_by_writer(tmp_path: Path, state: StateDb) -> None:
    writer = LanceStore(tmp_path, state, "m1", dim=DIM)
    writer.replace_rows(["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])])
    assert LanceStore(tmp_path, state, "m1", dim=None).count() == 1


def test_vectors_from_another_embedder_are_never_searched(tmp_path: Path, state: StateDb) -> None:
    writer = LanceStore(tmp_path, state, "m1", dim=DIM)
    writer.replace_rows(["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])])
    query = np.asarray([1, 0, 0, 0], np.float32)
    same = LanceStore(tmp_path, state, "M1:latest", dim=None)  # the same model, spelled out
    assert same.vectors_current()
    assert len(same.vector_search(lc.CHUNKS, query, ["path"])) == 1
    same.require_current_vectors()

    switched = LanceStore(tmp_path, state, "m2", dim=None)  # embedder changed, not rebuilt yet
    assert not switched.vectors_current()
    assert switched.vector_search(lc.CHUNKS, query, ["path"]) == []
    assert switched.fts_search(lc.CHUNKS, "x", ["path"]) is not None  # keywords still work
    with pytest.raises(lc.IndexRebuildingError, match="being rebuilt for m2"):
        switched.require_current_vectors()

    state.set_meta("model_id", "m2")  # the worker's rebuild has started over with m2
    assert switched.vectors_current()


def test_an_index_with_no_recorded_model_counts_as_current(tmp_path: Path, state: StateDb) -> None:
    assert LanceStore(tmp_path, state, "anything", dim=None).vectors_current()


def test_maintain_builds_vector_index_when_large(tmp_path: Path, state: StateDb) -> None:
    store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=1)
    rng = np.random.default_rng(0)
    rows = [chunk("a.py", f"t{i}", rng.random(DIM).tolist()) for i in range(300)]
    store.replace_rows(["a.py"], rows)
    store.maintain()
    kinds = {i.index_type for i in store.chunks.list_indices()}
    assert any(k != "FTS" for k in kinds)


class TestIndexMaintenance:
    def test_a_missing_keyword_index_is_repaired_and_the_vector_index_is_built(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=1)
        rows = [chunk(f"f{i}.py", f"word{i} alpha", [1.0, float(i), 0.0, 0.0]) for i in range(300)]
        store.replace_rows([r["path"] for r in rows], rows)
        table = store.chunks
        assert table is not None
        table.drop_index("search_text_idx")  # as if its creation had failed at table creation time
        assert not store._has_index_on(table, "search_text")
        store.maintain()
        assert store._has_index_on(table, "search_text")  # repaired
        assert store._has_index_on(table, lc.CHUNK_VECTOR)  # FTS no longer hides the need for it
        assert store.fts_search(lc.CHUNKS, "alpha", ["path"], "", 5)

    def test_lookup_columns_get_btree_indexes(self, tmp_path: Path, state: StateDb) -> None:
        store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=10_000)
        store.replace_rows(["a.py"], [chunk("a.py", "alpha", [1, 0, 0, 0])])
        store.replace_documents(["a.py"], [document("a.py", "alpha text", [1, 0, 0, 0])])
        store.maintain()
        store.maintain()  # idempotent: no error, no second index
        chunks, documents = store.chunks, store.documents
        assert chunks is not None
        assert documents is not None
        assert store._has_index_on(chunks, "chunk_hash")
        assert store._has_index_on(chunks, "path")
        assert store._has_index_on(documents, "path")
        assert store.vectors_for_hashes(["h-alpha"])  # lookups still answer through the index

    def test_old_table_versions_are_cleaned_up(
        self, tmp_path: Path, state: StateDb, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import timedelta

        store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=10_000)
        store.replace_rows(["a.py"], [chunk("a.py", "alpha", [1, 0, 0, 0])])
        requested: list[object] = []
        table = store.chunks
        assert table is not None
        original = table.optimize

        def spy(**kwargs: object) -> object:
            requested.append(kwargs.get("cleanup_older_than"))
            return original(**kwargs)

        monkeypatch.setattr(table, "optimize", spy)
        store.maintain()
        assert requested == [timedelta(hours=1)]

    def test_the_vector_index_is_rebuilt_when_the_table_has_doubled(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=1)

        def add(start: int, count: int) -> None:
            rows = [
                chunk(f"f{i}.py", f"word{i}", [1.0, float(i % 7), 0.0, 1.0])
                for i in range(start, start + count)
            ]
            store.replace_rows([r["path"] for r in rows], rows)

        add(0, 300)
        store.maintain()
        assert state.get_meta("vector_index_rows_chunks") == "300"
        add(300, 100)  # grew by a third: the index is kept
        store.maintain()
        assert state.get_meta("vector_index_rows_chunks") == "300"
        add(400, 300)  # now more than double
        store.maintain()
        assert state.get_meta("vector_index_rows_chunks") == "700"


V1_COLUMNS = [c for c in lc.CHUNK_COLUMNS if c not in ("content_chars", "name", "search_text")]


def build_v1_index(data_dir: Path, state: StateDb, rows: list[dict]) -> None:
    """An index as the first schema version wrote it: no derived columns, BM25 on ``text``."""
    import lancedb
    import pyarrow as pa
    from lancedb.index import FTS

    db = lancedb.connect(str(data_dir / lc.LANCE_DIRNAME))
    full = lc.chunk_schema(DIM)
    schema = pa.schema([full.field(lc.CHUNK_VECTOR)] + [full.field(c) for c in V1_COLUMNS])
    table = db.create_table(lc.CHUNKS, schema=schema)
    table.add([{k: v for k, v in row.items() if k in schema.names} for row in rows])
    table.create_index("text", config=FTS(with_position=True, stem=False))
    db.create_table(lc.DOCUMENTS, schema=lc.document_schema(DIM))
    state.set_meta("model_id", "m1")
    state.set_meta("dim", str(DIM))
    state.set_meta(lc.SCHEMA_VERSION_KEY, "1")


class TestSearchColumnsMigration:
    ROWS: ClassVar[list[dict]] = [
        chunk(
            "D:\\billing\\retryPayments.py",
            "p > D:\\billing\\retryPayments.py\ncharging cards",
            [1, 0, 0, 0],
        ),
        chunk(
            "D:\\scans\\scan_0042.pdf",
            "p > D:\\scans\\scan_0042.pdf > p.1\n  -- . --",
            [0, 1, 0, 0],
        ),
    ]

    def test_a_reader_of_an_old_index_keeps_searching_text(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        build_v1_index(tmp_path, state, self.ROWS)
        reader = LanceStore(tmp_path, state, "m1", dim=None)  # readers never migrate
        assert [r["path"] for r in reader.fts_search(lc.CHUNKS, "charging", ["path"])] == [
            "D:\\billing\\retryPayments.py"
        ]

    def test_the_content_floor_filters_new_indexes_and_is_ignored_on_old_ones(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        query = np.asarray([0, 1, 0, 0], np.float32)
        build_v1_index(tmp_path, state, self.ROWS)
        reader = LanceStore(tmp_path, state, "m1", dim=None)
        assert len(reader.vector_search(lc.CHUNKS, query, ["path"], min_content_chars=5)) == 2
        LanceStore(tmp_path, state, "m1", dim=DIM)  # the worker upgrades it
        upgraded = LanceStore(tmp_path, state, "m1", dim=None)
        kept = upgraded.vector_search(
            lc.CHUNKS, query, ["path"], "kind = 'code'", 10, min_content_chars=5
        )
        assert [r["path"] for r in kept] == ["D:\\billing\\retryPayments.py"]  # the scan is gone

    def test_v1_index_gains_backfilled_columns_and_stemmed_search(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        build_v1_index(tmp_path, state, self.ROWS)
        store = LanceStore(tmp_path, state, "m1", dim=DIM)
        assert state.get_meta(lc.SCHEMA_VERSION_KEY) == str(lc.LANCE_SCHEMA_VERSION)
        rows = {r["path"]: r for r in store.scan(lc.CHUNKS, lc.CHUNK_COLUMNS)}
        code = rows["D:\\billing\\retryPayments.py"]
        assert code["name"] == "retrypayments.py"
        assert code["content_chars"] == len("chargingcards")
        assert code["search_text"] == "retry payments\ncharging cards"
        assert rows["D:\\scans\\scan_0042.pdf"]["content_chars"] == 0
        # Stemmed, on the file-name words, and blind to directory names.
        assert store.fts_search(lc.CHUNKS, "charge", ["path"])
        assert store.fts_search(lc.CHUNKS, "payment", ["path"])
        assert store.fts_search(lc.CHUNKS, "billing", ["path"]) == []
        assert [i.columns for i in store.chunks.list_indices() if i.columns == ["text"]] == []

    def test_a_resumed_upgrade_does_not_add_the_columns_twice(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        build_v1_index(tmp_path, state, self.ROWS)
        LanceStore(tmp_path, state, "m1", dim=DIM)
        state.set_meta(lc.SCHEMA_VERSION_KEY, "1")  # as if it stopped before recording v2
        store = LanceStore(tmp_path, state, "m1", dim=DIM)
        assert store.count() == 2


class TestSchemaVersion:
    def test_new_tables_record_the_current_version(self, store: LanceStore, state: StateDb) -> None:
        assert state.get_meta(lc.SCHEMA_VERSION_KEY) == str(lc.LANCE_SCHEMA_VERSION)

    def test_search_results_carry_their_raw_scores(self, store: LanceStore) -> None:
        store.replace_rows(
            ["a.py", "b.py"],
            [chunk("a.py", "retry payments", [1, 0, 0, 0]), chunk("b.py", "garden", [0, 1, 0, 0])],
        )
        near = store.vector_search(lc.CHUNKS, np.asarray([1, 0, 0, 0], np.float32), ["path"])
        assert near[0]["path"] == "a.py"
        assert near[0]["_distance"] == pytest.approx(0.0, abs=1e-6)
        assert near[1]["_distance"] == pytest.approx(1.0, abs=1e-6)
        hits = store.fts_search(lc.CHUNKS, "payments", ["path"])
        assert hits[0]["path"] == "a.py"
        assert hits[0]["_score"] > 0

    def test_an_index_from_before_versions_runs_every_step_once(
        self, tmp_path: Path, state: StateDb, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        LanceStore(tmp_path, state, "m1", dim=DIM)
        state.set_meta(lc.SCHEMA_VERSION_KEY, "")  # as written before versions were recorded
        ran: list[str] = []
        steps = (lambda _s: ran.append("1->2"), lambda _s: ran.append("2->3"))
        monkeypatch.setattr(lc, "LANCE_MIGRATIONS", steps)
        monkeypatch.setattr(lc, "LANCE_SCHEMA_VERSION", 3)
        LanceStore(tmp_path, state, "m1", dim=DIM)
        LanceStore(tmp_path, state, "m1", dim=DIM)  # already current: nothing runs again
        assert ran == ["1->2", "2->3"]
        assert state.get_meta(lc.SCHEMA_VERSION_KEY) == "3"

    def test_an_interrupted_upgrade_resumes_at_the_failed_step(
        self, tmp_path: Path, state: StateDb, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        LanceStore(tmp_path, state, "m1", dim=DIM)
        state.set_meta(lc.SCHEMA_VERSION_KEY, "1")
        ran: list[str] = []

        def failing(_store: LanceStore) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(lc, "LANCE_MIGRATIONS", (lambda _s: ran.append("1->2"), failing))
        monkeypatch.setattr(lc, "LANCE_SCHEMA_VERSION", 3)
        with pytest.raises(OSError, match="disk full"):
            LanceStore(tmp_path, state, "m1", dim=DIM)
        assert state.get_meta(lc.SCHEMA_VERSION_KEY) == "2"  # the first step is kept
        monkeypatch.setattr(
            lc, "LANCE_MIGRATIONS", (lambda _s: ran.append("again"), lambda _s: ran.append("2->3"))
        )
        LanceStore(tmp_path, state, "m1", dim=DIM)
        assert ran == ["1->2", "2->3"]

    def test_an_index_from_a_newer_app_is_refused_for_reading_and_writing(
        self, tmp_path: Path, state: StateDb
    ) -> None:
        LanceStore(tmp_path, state, "m1", dim=DIM)
        state.set_meta(lc.SCHEMA_VERSION_KEY, str(lc.LANCE_SCHEMA_VERSION + 1))
        with pytest.raises(lc.IndexSchemaError, match="newer"):
            LanceStore(tmp_path, state, "m1", dim=DIM)
        with pytest.raises(lc.IndexSchemaError):
            LanceStore(tmp_path, state, "m1")  # read-only (search)
