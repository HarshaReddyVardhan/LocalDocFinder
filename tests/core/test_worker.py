from pathlib import Path

import pytest
from tests.core.conftest import Env

from vector_embed import worker
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.store.lance import CHUNKS, DOCUMENTS
from vector_embed.worker import WorkerOptions, WorkerParts, run_worker

RESUME = (
    "Jane Doe\nSummary\nBackend engineer\nWork Experience\nBuilt payment systems in Python\n"
    "Education\nBSc Computer Science\nSkills\nPython SQL\nProjects\nSearch engine\n"
)


class FakeGate:
    """Allows ``allowed`` checks, then refuses with ``reason``."""

    def __init__(self, allowed: int | None = None, reason: str = "unplugged") -> None:
        self.allowed = allowed
        self.reason = reason
        self.calls: list[tuple[bool, bool]] = []

    def worker_may_continue(
        self, allow_battery: bool = False, respect_activity: bool = True
    ) -> tuple[bool, str]:
        self.calls.append((allow_battery, respect_activity))
        if self.allowed is not None and len(self.calls) > self.allowed:
            return False, self.reason
        return True, ""


class Unloader:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> None:
        self.count += 1


def make_parts(
    env: Env, gate: FakeGate | None = None, unload: Unloader | None = None
) -> WorkerParts:
    return WorkerParts(
        settings=env.settings,
        state=env.state,
        store=env.store,
        embedder=env.embedder,
        extractors=env.extractors,
        projects=env.projects,
        scope=env.scope,
        classifier=env.classifier,
        gate=gate or FakeGate(),
        unload=unload or Unloader(),
    )


def write(env: Env, rel: str, content: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_drains_the_queue_and_unloads_the_model(env: Env) -> None:
    for i in range(3):
        env.state.enqueue(write(env, f"f{i}.txt", f"file {i} content\n" * 20), delay=0)
    unload = Unloader()
    assert run_worker(make_parts(env, unload=unload), WorkerOptions()) == 0
    assert env.state.queue_size() == 0
    assert env.state.manifest_count() == 3
    assert env.store.count(CHUNKS) >= 3
    assert unload.count == 1


def test_debounced_items_wait_unless_now(env: Env) -> None:
    env.state.enqueue(write(env, "a.txt", "alpha text\n" * 20), delay=3600)
    run_worker(make_parts(env), WorkerOptions())
    assert env.state.queue_size() == 1
    run_worker(make_parts(env), WorkerOptions(now=True))
    assert env.state.queue_size() == 0


def test_does_not_start_on_battery(env: Env) -> None:
    env.state.enqueue(write(env, "a.txt", "alpha text\n" * 20), delay=0)
    unload = Unloader()
    gate = FakeGate(allowed=0)
    assert run_worker(make_parts(env, gate, unload), WorkerOptions()) == 0
    assert env.state.queue_size() == 1
    assert env.store.count(CHUNKS) == 0
    assert gate.calls == [(False, False)]
    assert unload.count == 0


def test_unplugging_mid_run_commits_progress_and_keeps_the_queue(env: Env) -> None:
    env.settings = env.settings.model_copy(
        update={"chunking": env.settings.chunking.model_copy(update={"worker_batch_files": 1})}
    )
    for i in range(4):
        env.state.enqueue(write(env, f"f{i}.txt", f"file {i} content\n" * 20), delay=0)
    unload = Unloader()
    # initial check + a few stop_checks pass, then power is lost
    gate = FakeGate(allowed=6)
    run_worker(make_parts(env, gate, unload), WorkerOptions())
    done = env.state.manifest_count()
    assert 0 < done < 4
    assert env.state.queue_size() == 4 - done
    assert unload.count == 1


def test_no_maintenance_after_stopping_for_battery(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[str] = []
    monkeypatch.setattr(env.store, "maintain", lambda: ran.append("maintain"))
    monkeypatch.setattr(worker.Indexer, "assign_version_groups", lambda _self: ran.append("groups"))
    for i in range(3):
        env.state.enqueue(write(env, f"g{i}.txt", f"file {i} content " * 40), delay=0)
    run_worker(make_parts(env, FakeGate(allowed=3)), WorkerOptions())
    assert env.state.queue_size() > 0  # it did stop early
    assert ran == []


def test_maintenance_runs_after_a_complete_pass(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []
    monkeypatch.setattr(env.store, "maintain", lambda: ran.append("maintain"))
    env.state.enqueue(write(env, "h.txt", "content " * 40), delay=0)
    run_worker(make_parts(env), WorkerOptions(now=True))
    assert ran == ["maintain"]


def test_allow_battery_and_activity_flags_reach_the_gate(env: Env) -> None:
    gate = FakeGate()
    run_worker(make_parts(env, gate), WorkerOptions(allow_battery=True, now=True))
    assert gate.calls[0] == (True, False)
    assert all(call == (True, False) for call in gate.calls)
    gate = FakeGate()
    run_worker(make_parts(env, gate), WorkerOptions())
    assert gate.calls[1:] and all(call == (False, True) for call in gate.calls[1:])


def test_path_option_scans_and_indexes_without_a_queue(env: Env) -> None:
    write(env, "one/a.txt", "alpha text\n" * 20)
    write(env, "two/b.txt", "bravo text\n" * 20)
    run_worker(make_parts(env), WorkerOptions(now=True, paths=(str(env.root / "one"),)))
    assert env.state.manifest_count() == 1
    assert env.state.get_meta(worker.LAST_RECONCILE_KEY) is None  # partial scans do not count


def test_reconcile_option_records_the_time(env: Env) -> None:
    write(env, "a.txt", "alpha text\n" * 20)
    run_worker(make_parts(env), WorkerOptions(now=True, reconcile=True))
    assert env.state.manifest_count() == 1
    assert env.state.get_meta(worker.LAST_RECONCILE_KEY) is not None


def test_interrupted_reconcile_is_not_recorded(env: Env) -> None:
    write(env, "a.txt", "alpha text\n" * 20)
    gate = FakeGate(allowed=1)  # passes the start check, then stops during the scan
    run_worker(make_parts(env, gate), WorkerOptions(reconcile=True))
    assert env.state.get_meta(worker.LAST_RECONCILE_KEY) is None


def test_provider_failure_returns_2_and_backs_off(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(env, "a.txt", "alpha text\n" * 20)
    env.state.enqueue(path, delay=0)

    def boom(*_a: object, **_k: object) -> None:
        raise ProviderError("ollama down")

    monkeypatch.setattr(env.embedder, "embed", boom)
    unload = Unloader()
    assert run_worker(make_parts(env, unload=unload), WorkerOptions()) == worker.EXIT_PROVIDER
    assert env.state.claim(10) == []  # backed off for ten minutes
    assert env.state.queue_size() == 1
    assert unload.count == 1


def test_limit_stops_after_n_files(env: Env) -> None:
    env.settings = env.settings.model_copy(
        update={"chunking": env.settings.chunking.model_copy(update={"worker_batch_files": 1})}
    )
    for i in range(4):
        env.state.enqueue(write(env, f"f{i}.txt", f"file {i} content\n" * 20), delay=0)
    run_worker(make_parts(env), WorkerOptions(limit=2))
    assert env.state.manifest_count() == 2


def test_version_groups_are_assigned_after_indexing(env: Env) -> None:
    for name in ("Resume_v1.txt", "Resume_final.txt"):
        env.state.enqueue(write(env, name, RESUME * 4), delay=0)
    run_worker(make_parts(env), WorkerOptions())
    groups = {r["version_group"] for r in env.store.scan(DOCUMENTS, ["version_group"])}
    assert len(groups) == 1
    assert "" not in groups


def test_empty_queue_is_a_clean_run(env: Env) -> None:
    unload = Unloader()
    assert run_worker(make_parts(env, unload=unload), WorkerOptions()) == 0
    assert unload.count == 1


class TestCli:
    def test_parse_args(self) -> None:
        args = worker.parse_args(
            [
                "--now",
                "--path",
                "a",
                "--path",
                "b",
                "--reconcile",
                "--allow-battery",
                "--limit",
                "3",
            ]
        )
        assert args.now
        assert args.path == ["a", "b"]
        assert args.limit == 3

    def test_main_runs_the_worker_with_overrides(
        self, env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def fake_build(settings: object, _state: object, _gate: object) -> WorkerParts:
            seen["settings"] = settings
            return make_parts(env)

        monkeypatch.setattr(worker, "build_parts", fake_build)
        monkeypatch.setattr(worker, "build_gate", lambda _s, _st: FakeGate())
        monkeypatch.setattr(worker, "load_settings", lambda: env.settings)
        monkeypatch.setattr(worker, "configure_logging", lambda *_a, **_k: None)
        data_dir = tmp_path / "other-data"
        code = worker.main(["--now", "--model", "bge-m3", "--data-dir", str(data_dir)])
        assert code == 0
        settings = seen["settings"]
        assert settings.embedding.model == "bge-m3"  # type: ignore[attr-defined]
        assert settings.storage.data_dir == data_dir  # type: ignore[attr-defined]

    def test_second_worker_exits_quietly(self, env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
        from vector_embed.core.process import single_instance

        monkeypatch.setattr(worker, "load_settings", lambda: env.settings)
        monkeypatch.setattr(worker, "configure_logging", lambda *_a, **_k: None)

        def must_not_build(*_args: object) -> WorkerParts:
            raise AssertionError("second worker must not start")

        monkeypatch.setattr(worker, "build_parts", must_not_build)
        with single_instance("worker", env.data_dir):
            assert worker.main([]) == 0

    def test_main_checks_the_gate_before_touching_the_model_server(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def must_not_build(*_args: object) -> WorkerParts:
            raise AssertionError("the model server was used while on battery")

        monkeypatch.setattr(worker, "load_settings", lambda: env.settings)
        monkeypatch.setattr(worker, "configure_logging", lambda *_a, **_k: None)
        monkeypatch.setattr(worker, "build_gate", lambda _s, _st: FakeGate(allowed=0))
        monkeypatch.setattr(worker, "build_parts", must_not_build)
        assert worker.main([]) == 0

    def test_main_reports_an_unreachable_model_server(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(*_args: object) -> WorkerParts:
            raise ProviderError("ollama unavailable")

        monkeypatch.setattr(worker, "load_settings", lambda: env.settings)
        monkeypatch.setattr(worker, "configure_logging", lambda *_a, **_k: None)
        monkeypatch.setattr(worker, "build_gate", lambda _s, _st: FakeGate())
        monkeypatch.setattr(worker, "build_parts", down)
        assert worker.main([]) == worker.EXIT_PROVIDER

    def test_a_half_built_worker_unloads_the_embedder(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unloaded: list[int] = []

        class Provider:
            def unload_embedder(self) -> None:
                unloaded.append(1)

        def boom(*_args: object) -> None:
            raise ProviderError("embed failed")

        monkeypatch.setattr(worker.runtime, "build_scope", lambda _s: env.scope)
        monkeypatch.setattr(worker.runtime, "build_provider", lambda _s: Provider())
        monkeypatch.setattr(worker.runtime, "open_store", lambda *_a: env.store)
        monkeypatch.setattr(worker.runtime, "build_extractors", lambda *_a: env.extractors)
        monkeypatch.setattr(worker.runtime, "build_projects", lambda *_a: env.projects)
        monkeypatch.setattr(worker.runtime, "load_prototypes", boom)
        with pytest.raises(ProviderError):
            worker.build_parts(env.settings, env.state, FakeGate())
        assert unloaded == [1]
