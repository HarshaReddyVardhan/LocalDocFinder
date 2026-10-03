import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from vector_embed.core.store import sqlite as sq
from vector_embed.core.store.sqlite import FAILED_HASH, ManifestEntry, StateDb


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db(tmp_path: Path, clock: FakeClock) -> StateDb:
    with StateDb(tmp_path, clock=clock) as database:
        yield database  # type: ignore[misc]


def test_schema_version_is_stamped(tmp_path: Path) -> None:
    with StateDb(tmp_path):
        pass
    raw = sqlite3.connect(tmp_path / sq.STATE_FILENAME)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == sq.SCHEMA_VERSION
    raw.close()


def test_reopen_keeps_data(tmp_path: Path) -> None:
    with StateDb(tmp_path) as first:
        first.set_meta("k", "v")
    with StateDb(tmp_path) as second:
        assert second.get_meta("k") == "v"


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    with StateDb(tmp_path):
        pass
    raw = sqlite3.connect(tmp_path / sq.STATE_FILENAME)
    raw.execute(f"PRAGMA user_version={sq.SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(sq.StateError):
        StateDb(tmp_path)


def test_migrations_run_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with StateDb(tmp_path):
        pass

    def add_table(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE extra(x INTEGER)")

    monkeypatch.setattr(sq, "MIGRATIONS", (*sq.MIGRATIONS, add_table))
    monkeypatch.setattr(sq, "SCHEMA_VERSION", sq.SCHEMA_VERSION + 1)
    with StateDb(tmp_path) as upgraded:
        assert upgraded._one("SELECT COUNT(*) FROM extra") == (0,)
    raw = sqlite3.connect(tmp_path / sq.STATE_FILENAME)
    assert raw.execute("PRAGMA user_version").fetchone()[0] == sq.SCHEMA_VERSION
    raw.close()


def make_v1_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A database as released builds created it: case-sensitive paths, no claim index."""
    with monkeypatch.context() as patch:
        patch.setattr(sq, "MIGRATIONS", ())
        patch.setattr(sq, "SCHEMA_VERSION", 1)
        with StateDb(tmp_path) as legacy:
            legacy._run("INSERT INTO manifest VALUES('D:/Docs/A.txt',1,1,'h1',100)")
            legacy._run("INSERT INTO manifest VALUES('d:/docs/a.txt',2,2,'h2',200)")
            legacy._run("INSERT INTO manifest VALUES('D:/Docs/B.txt',3,3,'h3',300)")
            legacy._run("INSERT INTO queue VALUES('D:/Docs/C.txt','upsert',0,0,1,0)")
            legacy._run("INSERT INTO queue VALUES('d:/docs/c.txt','delete',0,0,2,0)")


class TestCaseInsensitivePaths:
    def test_upgrading_collapses_case_duplicates_and_remembers_the_dropped_spelling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_v1_database(tmp_path, monkeypatch)
        with StateDb(tmp_path) as db:
            assert db.manifest_count() == 2  # the A.txt pair became one
            entry = db.manifest_get("D:/DOCS/a.TXT")
            assert entry is not None
            assert entry.content_hash == "h2"  # the most recently indexed row won
            assert db.queue_size() == 1
            assert db.claim(10, ignore_debounce=True)[0].op == "delete"  # the newest request won
            assert db.take_stale_paths() == ["D:/Docs/A.txt"]
            assert db.take_stale_paths() == []  # handed out once

    def test_new_paths_match_regardless_of_case(self, db: StateDb) -> None:
        db.manifest_set("D:/Work/Plan.md", 1, 1, "h")
        db.manifest_set("d:/work/plan.md", 2, 2, "h2")
        assert db.manifest_count() == 1
        assert db.canonical_path("D:/WORK/PLAN.MD") == "D:/Work/Plan.md"  # first spelling
        assert db.canonical_path("D:/Work/new.md") == "D:/Work/new.md"  # unknown stays
        db.enqueue("D:/Work/X.md")
        db.enqueue("d:/work/x.md")
        assert db.queue_size() == 1

    def test_the_claim_index_exists(self, db: StateDb) -> None:
        names = {r[1] for r in db._all("PRAGMA index_list('queue')")}
        assert "queue_claim" in names

    def test_manifest_all_is_keyed_in_lower_case(self, db: StateDb) -> None:
        db.manifest_set("D:/Mixed/Case.TXT", 5, 6, "h")
        assert db.manifest_all() == {os.path.normcase("d:/mixed/case.txt"): (5, 6)}

    def test_two_processes_opening_a_fresh_database_do_not_collide(self, tmp_path: Path) -> None:
        import threading

        errors: list[BaseException] = []

        def open_it() -> None:
            try:
                with StateDb(tmp_path):
                    pass
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=open_it) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []


class TestTransactions:
    def test_a_failure_inside_a_batch_rolls_everything_back(self, db: StateDb) -> None:
        def models() -> Iterator[dict[str, object]]:
            yield {"name": "good"}
            raise RuntimeError("boom")

        db.replace_models("ollama", [{"name": "kept"}])
        with pytest.raises(RuntimeError):
            db.replace_models("ollama", models())  # type: ignore[arg-type]
        assert [m["name"] for m in db.list_models("ollama")] == ["kept"]
        db.enqueue("a")  # and the connection is usable: no transaction was left open
        assert db.queue_size() == 1


class TestMeta:
    def test_default_and_overwrite(self, db: StateDb) -> None:
        assert db.get_meta("x") is None
        assert db.get_meta("x", "d") == "d"
        db.set_meta("x", "1")
        db.set_meta("x", "2")
        assert db.get_meta("x") == "2"


class TestManifest:
    def test_roundtrip_and_update(self, db: StateDb) -> None:
        assert db.manifest_get("a") is None
        db.manifest_set("a", 1, 2, "h1")
        assert db.manifest_get("a") == ManifestEntry(1, 2, "h1")
        db.manifest_set("a", 3, 4, "h2")
        assert db.manifest_get("a") == ManifestEntry(3, 4, "h2")
        assert db.manifest_all() == {"a": (3, 4)}
        assert db.manifest_count() == 1

    def test_delete_and_clear(self, db: StateDb) -> None:
        db.manifest_set("a", 1, 1, "h")
        db.manifest_set("b", 1, 1, "h")
        db.manifest_delete("a")
        assert db.manifest_count() == 1
        db.manifest_clear()
        assert db.manifest_count() == 0

    def test_under_is_case_insensitive_and_directory_bounded(self, db: StateDb) -> None:
        base = "D:" + os.sep + "Proj"
        db.manifest_set(base + os.sep + "a.py", 1, 1, "h")
        db.manifest_set(base.lower() + os.sep + "b.py", 1, 1, "h")
        db.manifest_set(base + "-other" + os.sep + "c.py", 1, 1, "h")
        assert sorted(db.manifest_under(base)) == sorted(
            [base + os.sep + "a.py", base.lower() + os.sep + "b.py"]
        )


class TestQueue:
    def test_debounce_hides_fresh_events(self, db: StateDb, clock: FakeClock) -> None:
        db.enqueue("a", delay=30)
        assert db.queue_size() == 1
        assert db.queue_size(due_only=True) == 0
        assert db.claim(10) == []
        clock.now += 31
        assert [i.path for i in db.claim(10)] == ["a"]
        assert db.queue_size(due_only=True) == 1

    def test_ignore_debounce(self, db: StateDb) -> None:
        db.enqueue("a", delay=999)
        assert len(db.claim(10, ignore_debounce=True)) == 1

    def test_priority_then_arrival_order(self, db: StateDb) -> None:
        db.enqueue("low", priority=1)
        db.enqueue("high", priority=9)
        db.enqueue("low2", priority=1)
        assert [i.path for i in db.claim(10)] == ["high", "low", "low2"]

    def test_requeue_while_processing_keeps_the_row(self, db: StateDb) -> None:
        db.enqueue("a")
        (item,) = db.claim(10)
        db.enqueue("a")  # file changed again mid-processing; new seq
        db.done([(item.path, item.seq)])
        assert db.queue_size() == 1
        (again,) = db.claim(10)
        db.done([(again.path, again.seq)])
        assert db.queue_size() == 0

    def test_enqueue_many_and_op(self, db: StateDb) -> None:
        assert db.enqueue_many([("a", "upsert", 1.0), ("b", "delete", 0.0)]) == 2
        ops = {i.path: i.op for i in db.claim(10)}
        assert ops == {"a": "upsert", "b": "delete"}

    def test_enqueue_many_rolls_back_on_error(self, db: StateDb) -> None:
        def items():  # type: ignore[no-untyped-def]
            yield ("a", "upsert", 0.0)
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            db.enqueue_many(items())
        assert db.queue_size() == 0

    def test_fail_backs_off_then_gives_up(
        self, db: StateDb, clock: FakeClock, tmp_path: Path
    ) -> None:
        target = tmp_path / "bad.pdf"
        target.write_bytes(b"x" * 10)
        path = str(target)
        db.enqueue(path)
        for _ in range(3):
            clock.now += 1000
            assert db.claim(1)
            assert db.queue_size(due_only=True) == 1
            db.fail(path, delay=10)
        clock.now += 1000
        assert db.claim(1) == []
        assert db.queue_size(due_only=True) == 0  # a dead row is not work for the watcher
        assert db.queue_size() == 0  # it left the queue
        entry = db.manifest_get(path)
        assert entry is not None
        assert entry.content_hash == sq.FAILED_HASH  # remembered as seen, until it changes
        assert (entry.mtime_ns, entry.size) == (target.stat().st_mtime_ns, 10)
        db.enqueue(path)  # an explicit re-queue (the file changed) gets a fresh set of attempts
        assert db.claim(1)

    def test_dead_rows_do_not_count_as_due_while_failing(
        self, db: StateDb, clock: FakeClock
    ) -> None:
        db.enqueue("a")
        db.enqueue("b")
        assert db.queue_size(due_only=True) == 2
        db.fail("a", delay=10)
        assert db.queue_size(due_only=True) == 1  # "a" is backing off
        clock.now += 20
        assert db.queue_size(due_only=True) == 2

    def test_giving_up_on_a_vanished_file_clears_it_everywhere(
        self, db: StateDb, clock: FakeClock
    ) -> None:
        db.manifest_set("gone.txt", 1, 1, "old")
        db.enqueue("gone.txt")
        for _ in range(3):
            db.fail("gone.txt", delay=0)
        assert db.manifest_get("gone.txt") is None
        assert db.queue_size() == 0


class TestLocks:
    def test_exclusive_until_expiry(self, db: StateDb, clock: FakeClock) -> None:
        assert db.acquire_lock("chat", "me", 60)
        assert db.lock_held("chat")
        assert not db.acquire_lock("chat", "other", 60)
        assert db.acquire_lock("chat", "me", 60)  # refresh by the owner
        clock.now += 61
        assert not db.lock_held("chat")
        assert db.acquire_lock("chat", "other", 60)

    def test_release_only_by_owner(self, db: StateDb) -> None:
        db.acquire_lock("chat", "me", 60)
        db.release_lock("chat", "other")
        assert db.lock_held("chat")
        db.release_lock("chat", "me")
        assert not db.lock_held("chat")


class TestChat:
    def test_session_roundtrip(self, db: StateDb, clock: FakeClock) -> None:
        sid = db.create_session("JD match", {"pinned": ["a.pdf"]})
        db.add_message(sid, "user", "hi")
        clock.now += 5
        db.add_message(sid, "assistant", "hello")
        assert [(m.role, m.content) for m in db.messages(sid)] == [
            ("user", "hi"),
            ("assistant", "hello"),
        ]
        assert db.session_context(sid) == {"pinned": ["a.pdf"]}
        (session,) = db.sessions()
        assert session.title == "JD match"
        assert session.updated_at == clock.now

    def test_missing_session_context_is_empty(self, db: StateDb) -> None:
        assert db.session_context(999) == {}

    def test_sessions_newest_first(self, db: StateDb, clock: FakeClock) -> None:
        db.create_session("old")
        clock.now += 10
        db.create_session("new")
        assert [s.title for s in db.sessions()] == ["new", "old"]


class TestModels:
    def test_replace_and_list(self, db: StateDb) -> None:
        db.replace_models(
            "ollama",
            [
                {"name": "a", "size_bytes": 5, "capabilities": ["completion", "tools"]},
                {"name": "b", "context_length": 8192, "capabilities": ["embedding"]},
            ],
        )
        listed = db.list_models("ollama")
        assert [m["name"] for m in listed] == ["a", "b"]
        assert listed[0]["capabilities"] == ["completion", "tools"]
        assert listed[1]["context_length"] == 8192
        db.replace_models("ollama", [{"name": "c"}])
        assert [m["name"] for m in db.list_models()] == ["c"]
        assert db.list_models("other") == []


class TestUsage:
    def test_totals_and_spend(self, db: StateDb, clock: FakeClock) -> None:
        db.record_usage("openrouter", "m1", 100, 10, 0.5)
        db.record_usage("openrouter", "m1", 50, 5, 0.25)
        clock.now += 100
        db.record_usage("openai", "m2", 1, 1, 1.0)
        totals = {(t.provider, t.model): t for t in db.usage_totals()}
        assert totals[("openrouter", "m1")].prompt_tokens == 150
        assert totals[("openrouter", "m1")].cost_usd == pytest.approx(0.75)
        assert db.spend_since(0) == pytest.approx(1.75)
        assert db.spend_since(1050) == pytest.approx(1.0)
        assert db.spend_since(0, provider="openrouter") == pytest.approx(0.75)
        assert db.usage_totals(since=1050)[0].provider == "openai"


class TestChatHistoryCap:
    def test_a_long_chat_keeps_only_the_newest_messages(
        self, db: StateDb, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sq, "MAX_STORED_MESSAGES", 5)
        session = db.create_session("long chat")
        other = db.create_session("other chat")
        db.add_message(other, "user", "untouched")
        for i in range(12):
            db.add_message(session, "user", f"message {i}")
        kept = [m.content for m in db.messages(session)]
        assert kept == [f"message {i}" for i in range(7, 12)]  # the newest five, in order
        assert [m.content for m in db.messages(other)] == ["untouched"]  # others are not trimmed


def test_manifest_list_filters_and_marks_failures(tmp_path: Path) -> None:
    with StateDb(tmp_path) as db:
        db.manifest_set("D:/Work/Plan.md", 1, 2048, "h")
        db.manifest_set("D:/Work/Broken.pdf", 1, 10, FAILED_HASH)
        assert {f.path for f in db.manifest_list()} == {"D:/Work/Plan.md", "D:/Work/Broken.pdf"}
        assert [f.path for f in db.manifest_list("plan")] == ["D:/Work/Plan.md"]
        failed = db.manifest_list(failed_only=True)
        assert [(f.path, f.failed) for f in failed] == [("D:/Work/Broken.pdf", True)]
        assert len(db.manifest_list(limit=1)) == 1
