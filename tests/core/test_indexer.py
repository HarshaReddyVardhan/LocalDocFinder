import os
from pathlib import Path

import numpy as np
import pytest
from tests.core.conftest import Env
from tests.core.fakes import DIM

from vector_embed.core.providers.ollama import Interrupted
from vector_embed.core.reconcile import reconcile
from vector_embed.core.store.lance import CHUNKS, DOCUMENTS
from vector_embed.core.store.sqlite import QueueItem

PY_A = (
    "def alpha():\n    return 'alpha value ' * 3\n\n\ndef beta():\n    return 'beta value ' * 3\n"
)
RESUME = (
    "Jane Doe\nSummary\nBackend engineer\nWork Experience\nBuilt payment systems in Python\n"
    "Education\nBSc Computer Science\nSkills\nPython SQL\nProjects\nSearch engine\n"
)


def write(env: "Env", rel: str, content: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def upsert(env: "Env", *paths: str, force: bool = False) -> list[tuple[str, int]]:
    return env.indexer.process([QueueItem(p, "upsert", i) for i, p in enumerate(paths)], force)


def test_indexes_file_into_chunks_documents_and_manifest(env: "Env") -> None:
    path = write(env, "proj/pyproject.toml", "[project]\n")
    path = write(env, "proj/a.py", PY_A)
    finished = upsert(env, path)
    assert finished == [(path, 0)]
    assert env.store.count(CHUNKS) >= 2
    assert env.store.count(DOCUMENTS) == 1
    assert env.state.manifest_get(path) is not None
    rows = env.store.scan(CHUNKS, ["project", "kind", "symbol", "ext"], f"path = '{path}'")
    assert {r["project"] for r in rows} == {"proj"}
    symbols = " ".join(r["symbol"] for r in rows)
    assert "alpha" in symbols
    assert "beta" in symbols
    assert env.indexer.stats.files == 1


def test_unchanged_file_is_skipped_without_embedding(env: "Env") -> None:
    path = write(env, "a.txt", "hello world " * 10)
    upsert(env, path)
    embedded = len(env.embedder.embedded_texts)
    upsert(env, path)
    assert len(env.embedder.embedded_texts) == embedded
    assert env.indexer.stats.skipped == 1


def test_touched_but_identical_file_only_updates_the_manifest(env: "Env") -> None:
    path = write(env, "a.txt", "hello world " * 10)
    upsert(env, path)
    embedded = len(env.embedder.embedded_texts)
    stat = Path(path).stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    upsert(env, path)
    assert len(env.embedder.embedded_texts) == embedded
    assert env.state.manifest_get(path).mtime_ns == stat.st_mtime_ns + 5_000_000_000  # type: ignore[union-attr]


def test_editing_one_function_reembeds_only_that_chunk(env: "Env") -> None:
    path = write(env, "proj/pyproject.toml", "[project]\n")
    path = write(env, "proj/a.py", PY_A)
    env.indexer.index_paths([path])
    before = env.indexer.stats.embedded
    Path(path).write_text(PY_A.replace("beta value", "gamma value"), encoding="utf-8")
    env.indexer.index_paths([path])
    # the outline chunk (it lists symbols) and the changed function are new; alpha is reused
    assert env.indexer.stats.embedded - before <= 2
    assert env.indexer.stats.reused >= 1
    symbols = " ".join(r["symbol"] for r in env.store.scan(CHUNKS, ["symbol"], f"path = '{path}'"))
    assert "alpha" in symbols


def test_identical_code_in_another_project_reuses_vectors(env: "Env") -> None:
    write(env, "p1/pyproject.toml", "[project]\n")
    write(env, "p2/pyproject.toml", "[project]\n")
    one = write(env, "p1/lib.py", PY_A)
    two = write(env, "p2/lib.py", PY_A)
    env.indexer.index_paths([one])
    embedded = env.indexer.stats.embedded
    env.indexer.index_paths([two])
    code_embeds = [t for t in env.embedder.embedded_texts if "alpha value" in t]
    assert len(code_embeds) == 1
    assert env.indexer.stats.embedded - embedded <= 1  # only the project-specific outline


def test_delete_removes_rows_and_manifest(env: "Env") -> None:
    path = write(env, "a.txt", "hello world " * 10)
    upsert(env, path)
    finished = env.indexer.process([QueueItem(path, "delete", 7)])
    assert finished == [(path, 7)]
    assert env.store.count(CHUNKS) == 0
    assert env.store.count(DOCUMENTS) == 0
    assert env.state.manifest_get(path) is None
    assert env.indexer.stats.deleted == 1


def test_vanished_or_now_invalid_files_are_deleted(env: "Env") -> None:
    gone = str(env.root / "ghost.txt")
    secret = write(env, ".env", "TOKEN=1")
    finished = env.indexer.process([QueueItem(gone, "upsert", 1), QueueItem(secret, "upsert", 2)])
    assert sorted(finished) == sorted([(gone, 1), (secret, 2)])
    assert env.indexer.stats.deleted == 2


def test_extract_errors_are_recorded_and_rows_dropped(env: "Env") -> None:
    path = write(env, "a.txt", "text " * 20)
    upsert(env, path)
    Path(path).write_bytes(b"binary\x00data")  # now unreadable as text
    upsert(env, path)
    assert env.store.count(CHUNKS) == 0
    assert env.indexer.stats.errors == 1
    assert env.state.manifest_get(path) is not None  # not retried until it changes


def test_unexpected_error_marks_the_item_failed(
    env: "Env", monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(env, "a.txt", "text " * 20)
    env.state.enqueue(path)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(env.extractors, "extract", boom)
    assert upsert(env, path) == []
    assert env.indexer.stats.errors == 1
    (row,) = env.state._all("SELECT attempts FROM queue")
    assert row[0] == 1


def test_stop_check_leaves_remaining_items_queued(env: "Env") -> None:
    one = write(env, "a.txt", "one " * 20)
    two = write(env, "b.txt", "two " * 20)
    env.indexer.stop_check = lambda: True
    assert upsert(env, one, two) == []
    assert env.store.count(CHUNKS) == 0


def test_interrupt_while_embedding_commits_nothing(env: "Env") -> None:
    path = write(env, "a.txt", "one " * 20)
    calls = iter([False, True])  # passes the per-file check, fires inside the embedder
    env.indexer.stop_check = lambda: next(calls)
    with pytest.raises(Interrupted):
        upsert(env, path)
    assert env.store.count(CHUNKS) == 0
    assert env.state.manifest_get(path) is None


def test_ai_notes_are_tagged(env: "Env") -> None:
    path = write(env, "notes/.claude/plans/p.md", "# Plan\n\nstep one text here\n")
    upsert(env, path)
    (row,) = env.store.scan(CHUNKS, ["kind", "source"], f"path = '{path}'")
    assert (row["kind"], row["source"]) == ("ai-note", "claude-plan")


def test_document_rows_have_type_text_and_unit_vector(env: "Env") -> None:
    path = write(env, "Resume_v1.txt", RESUME)
    upsert(env, path)
    (doc,) = env.store.scan(
        DOCUMENTS, ["doc_type", "title", "full_text", "doc_vector", "modified_at"]
    )
    assert doc["doc_type"] == "resume"
    assert doc["title"] == "Jane Doe"
    assert "payment systems" in doc["full_text"]
    assert np.linalg.norm(doc["doc_vector"]) == pytest.approx(1.0, abs=1e-4)
    assert len(doc["doc_vector"]) == DIM


def test_large_documents_keep_no_full_text(env: "Env") -> None:
    env.settings = env.settings.model_copy(
        update={"doctypes": env.settings.doctypes.model_copy(update={"full_text_max_chars": 50})}
    )
    env.indexer.settings = env.settings
    upsert(env, write(env, "long.txt", ("a line of words" + "\n") * 30))
    (doc,) = env.store.scan(DOCUMENTS, ["full_text"])
    assert doc["full_text"] == ""


def test_version_groups_are_assigned_for_near_identical_resumes(env: "Env") -> None:
    one = write(env, "Resume_v1.txt", RESUME * 4)
    two = write(env, "Resume_final.txt", RESUME * 4)
    other = write(env, "notes.txt", ("groceries and holiday plans" + "\n") * 20)
    env.indexer.index_paths([one, two, other])
    assert env.indexer.assign_version_groups() == 2
    docs = {
        Path(r["path"]).name: r["version_group"]
        for r in env.store.scan(DOCUMENTS, ["path", "version_group"])
    }
    assert docs["Resume_v1.txt"] == docs["Resume_final.txt"] != ""
    assert docs["notes.txt"] == ""


def test_version_groups_without_documents_table_rows(env: "Env") -> None:
    assert env.indexer.assign_version_groups() == 0


def test_index_paths_batches(env: "Env") -> None:
    paths = [write(env, f"f{i}.txt", f"file {i} " * 20) for i in range(5)]
    env.indexer.index_paths(paths, batch=2)
    assert env.indexer.stats.files == 5


def test_force_reindexes_unchanged_files(env: "Env") -> None:
    path = write(env, "a.txt", "text " * 20)
    upsert(env, path)
    upsert(env, path, force=True)
    assert env.indexer.stats.files == 2


class TestReconcile:
    def test_queues_new_changed_and_vanished(self, env: "Env") -> None:
        a = write(env, "a.txt", "alpha " * 10)
        b = write(env, "b.txt", "bravo " * 10)
        first = reconcile(env.state, env.projects, env.scope)
        assert first.queued == 2
        env.indexer.process(env.state.claim(10, ignore_debounce=True))
        env.state.done([(a, 0), (b, 0)])
        Path(a).write_text("alpha changed " * 10, encoding="utf-8")
        Path(b).unlink()
        write(env, "c.txt", "charlie " * 10)
        second = reconcile(env.state, env.projects, env.scope)
        ops = {(Path(i.path).name, i.op) for i in env.state.claim(10, ignore_debounce=True)}
        assert ops == {("a.txt", "upsert"), ("c.txt", "upsert"), ("b.txt", "delete")}
        assert (second.queued, second.deleted) == (2, 1)

    def test_unchanged_tree_queues_nothing(self, env: "Env") -> None:
        upsert(env, write(env, "a.txt", "alpha " * 10))
        result = reconcile(env.state, env.projects, env.scope)
        assert (result.queued, result.deleted) == (0, 0)

    def test_a_scan_that_spells_the_path_differently_changes_nothing(self, env: "Env") -> None:
        upsert(env, write(env, "Docs/Report.txt", "report " * 10))
        shouting = str(env.root).upper()  # Windows paths ignore case: the same files
        result = reconcile(env.state, env.projects, env.scope, roots=[shouting])
        assert (result.queued, result.deleted) == (0, 0)

    def test_a_case_only_rename_does_not_duplicate_search_rows(self, env: "Env") -> None:
        original = write(env, "Notes/Plan.txt", "plan words " * 10)
        upsert(env, original)
        before = env.store.count(CHUNKS)
        Path(original).write_text("plan words changed " * 10, encoding="utf-8")
        respelled = original.upper()  # a later event reports the file with another spelling
        env.state.enqueue(respelled, delay=0)
        env.indexer.process(env.state.claim(10, ignore_debounce=True))
        paths = {r["path"] for r in env.store.scan(CHUNKS, ["path"], limit=500)}
        assert len(paths) == 1  # one spelling in the index, not two copies of the file
        assert env.store.count(CHUNKS) >= before

    def test_stop_check_interrupts(self, env: "Env") -> None:
        write(env, "a.txt", "alpha " * 10)
        result = reconcile(env.state, env.projects, env.scope, stop_check=lambda: True)
        assert result.interrupted
        assert env.state.queue_size() == 0

    def test_files_outside_the_scanned_roots_are_left_alone(self, env: "Env") -> None:
        env.state.manifest_set(str(env.root.parent / "elsewhere" / "x.txt"), 1, 1, "h")
        assert reconcile(env.state, env.projects, env.scope).deleted == 0

    def test_explicit_roots(self, env: "Env") -> None:
        write(env, "one/a.txt", "a " * 10)
        write(env, "two/b.txt", "b " * 10)
        result = reconcile(env.state, env.projects, env.scope, roots=[str(env.root / "one")])
        assert result.queued == 1


def test_extractor_models_are_released_before_embedding(
    env: "Env", monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    monkeypatch.setattr(env.extractors, "release_models", lambda: order.append("release"))
    original = env.embedder.embed

    def embed(*args: object, **kwargs: object) -> np.ndarray:
        order.append("embed")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(env.embedder, "embed", embed)
    env.indexer.index_paths([write(env, "doc.md", "# Title\n\nsome words to embed " * 5)])
    assert order[0] == "release"  # the vision model leaves the GPU before the embedder loads
    assert "embed" in order


def test_credentials_pasted_into_a_note_never_reach_the_index(env: "Env") -> None:
    key = "sk-" + "abcdefghijklmnopqrstuvwxyz123456"
    path = write(
        env, "notes.md", f"# Setup\n\nOur key is {key} keep it safe.\n\nOther text here.\n"
    )
    env.indexer.index_paths([path])
    rows = env.store.scan(CHUNKS, ["text", "path"], limit=50)
    assert rows
    assert all(key not in row["text"] for row in rows)
    assert any("[SECRET REMOVED]" in row["text"] for row in rows)
    docs = env.store.scan(DOCUMENTS, ["full_text"], limit=5)
    assert all(key not in row["full_text"] for row in docs)


def test_ordinary_code_with_password_assignments_is_left_alone(env: "Env") -> None:
    path = write(
        env, "auth.py", "def login(request):\n    token = get_token(request)\n    return token\n"
    )
    env.indexer.index_paths([path])
    text = " ".join(r["text"] for r in env.store.scan(CHUNKS, ["text"], limit=50))
    assert "get_token(request)" in text


def test_a_locked_file_keeps_its_rows_instead_of_being_deleted(
    env: "Env", monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(env, "report.md", "# Report quarterly numbers for the team " * 5)
    env.indexer.index_paths([path])
    assert env.store.count(CHUNKS) > 0
    env.state.enqueue(path, delay=0)
    original = env.indexer.prepare

    def locked(p: str, seq: int = 0, force: bool = False) -> object:
        raise PermissionError("another process has the file open")

    monkeypatch.setattr(env.indexer, "prepare", locked)
    items = env.state.claim(10, ignore_debounce=True)
    finished = env.indexer.process(items)
    monkeypatch.setattr(env.indexer, "prepare", original)
    assert finished == []  # not marked done: it stays queued for a retry
    assert env.store.count(CHUNKS) > 0  # nothing was deleted
    assert env.state.manifest_get(path) is not None
    assert env.state.queue_size() == 1


def test_a_vanished_file_is_still_deleted(env: "Env") -> None:
    path = write(env, "temp.md", "# Temp some words here " * 5)
    env.indexer.index_paths([path])
    Path(path).unlink()
    env.state.enqueue(path, delay=0)
    env.indexer.process(env.state.claim(10, ignore_debounce=True))
    assert env.state.manifest_get(path) is None
