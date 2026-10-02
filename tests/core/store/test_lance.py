from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vector_embed.core.store import lance as lc
from vector_embed.core.store.lance import LanceStore, sql_quote
from vector_embed.core.store.sqlite import StateDb

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


def test_model_change_wipes_everything(tmp_path: Path, state: StateDb) -> None:
    first = LanceStore(tmp_path, state, "m1", dim=DIM)
    first.replace_rows(["a.py"], [chunk("a.py", "x", [1, 0, 0, 0])])
    state.manifest_set("a.py", 1, 1, "h")
    second = LanceStore(tmp_path, state, "m2", dim=DIM)
    assert second.count() == 0
    assert state.manifest_count() == 0
    assert state.get_meta("model_id") == "m2"


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


def test_maintain_builds_vector_index_when_large(tmp_path: Path, state: StateDb) -> None:
    store = LanceStore(tmp_path, state, "m1", dim=DIM, vector_index_min_rows=1)
    rng = np.random.default_rng(0)
    rows = [chunk("a.py", f"t{i}", rng.random(DIM).tolist()) for i in range(300)]
    store.replace_rows(["a.py"], rows)
    store.maintain()
    kinds = {i.index_type for i in store.chunks.list_indices()}
    assert any(k != "FTS" for k in kinds)
