import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.core.conftest import Env
from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
)

from vector_embed import watcher
from vector_embed.core import ollama_http
from vector_embed.core.process import request_stop
from vector_embed.watcher import ChangeHandler, Watcher, quick_reject

DEBOUNCE = 30


def queued(env: Env) -> dict[str, str]:
    return {i.path: i.op for i in env.state.claim(100, ignore_debounce=True)}


@pytest.fixture
def handler(env: Env) -> ChangeHandler:
    return ChangeHandler(env.state, env.projects, env.scope, env.settings)


def write(env: Env, rel: str, content: str = "x = 1\n") -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestChangeHandler:
    def test_modified_file_is_queued_with_debounce(self, handler: ChangeHandler, env: Env) -> None:
        path = write(env, "a.py")
        handler.on_modified(FileModifiedEvent(path))
        assert queued(env) == {path: "upsert"}
        assert env.state.claim(10) == []  # still debouncing
        assert env.state.queue_size() == 1

    def test_noise_and_secrets_are_not_queued(self, handler: ChangeHandler, env: Env) -> None:
        for rel in ("node_modules/x.js", "package-lock.json", ".env", "dist/app.min.js"):
            handler.on_created(FileCreatedEvent(write(env, rel)))
        ok = write(env, "ok.py")
        handler.on_created(FileCreatedEvent(ok))
        assert queued(env) == {ok: "upsert"}

    def test_gitignored_files_are_not_queued(self, handler: ChangeHandler, env: Env) -> None:
        write(env, "p/pyproject.toml", "[project]\n")
        write(env, "p/.gitignore", "gen.py\n")
        gen, keep = write(env, "p/gen.py"), write(env, "p/keep.py")
        handler.on_created(FileCreatedEvent(gen))
        handler.on_created(FileCreatedEvent(keep))
        assert queued(env) == {keep: "upsert"}

    def test_delete_is_queued_only_for_indexed_files(
        self, handler: ChangeHandler, env: Env
    ) -> None:
        indexed, other = write(env, "i.py"), write(env, "o.py")
        env.state.manifest_set(indexed, 1, 1, "h")
        handler.on_deleted(FileDeletedEvent(indexed))
        handler.on_deleted(FileDeletedEvent(other))
        assert queued(env) == {indexed: "delete"}
        assert env.state.claim(10)[0].op == "delete"  # deletes are not debounced

    def test_directory_delete_queues_every_indexed_file_below(
        self, handler: ChangeHandler, env: Env
    ) -> None:
        inside = str(env.root / "d" / "a.py")
        outside = str(env.root / "e" / "b.py")
        env.state.manifest_set(inside, 1, 1, "h")
        env.state.manifest_set(outside, 1, 1, "h")
        handler.on_deleted(DirDeletedEvent(str(env.root / "d")))
        assert queued(env) == {inside: "delete"}

    def test_move_deletes_the_old_path_and_queues_the_new(
        self, handler: ChangeHandler, env: Env
    ) -> None:
        old = str(env.root / "old.py")
        new = write(env, "new.py")
        env.state.manifest_set(old, 1, 1, "h")
        handler.on_moved(FileMovedEvent(old, new))
        assert queued(env) == {old: "delete", new: "upsert"}

    def test_directory_move_and_create_enqueue_the_tree(
        self, handler: ChangeHandler, env: Env
    ) -> None:
        old_file = str(env.root / "src" / "a.py")
        env.state.manifest_set(old_file, 1, 1, "h")
        new_file = write(env, "dst/a.py")
        handler.on_moved(DirMovedEvent(str(env.root / "src"), str(env.root / "dst")))
        deadline = time.time() + 5
        while new_file not in queued(env) and time.time() < deadline:
            time.sleep(0.05)
        assert queued(env) == {old_file: "delete", new_file: "upsert"}
        created = write(env, "fresh/b.py")
        handler.on_created(DirCreatedEvent(str(env.root / "fresh")))
        deadline = time.time() + 5
        while created not in queued(env) and time.time() < deadline:
            time.sleep(0.05)
        assert queued(env)[created] == "upsert"

    def test_tree_enqueue_failure_is_contained(
        self, handler: ChangeHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_roots: list[str]) -> None:
            raise OSError("denied")

        monkeypatch.setattr(handler.projects, "iter_files", boom)
        handler._enqueue_tree("anywhere")  # must not raise

    def test_byte_paths_are_decoded(self, handler: ChangeHandler, env: Env) -> None:
        path = write(env, "b.py")
        handler.on_modified(FileModifiedEvent(path.encode()))
        assert queued(env) == {path: "upsert"}

    def test_directory_modifications_are_ignored(self, handler: ChangeHandler, env: Env) -> None:
        handler.on_modified(DirModifiedEvent(str(env.root)))
        assert queued(env) == {}


def test_quick_reject() -> None:
    blocked = frozenset({"node_modules", ".git"})
    assert quick_reject("D:\\p\\node_modules\\x\\a.js".replace("\\", __import__("os").sep), blocked)
    assert not quick_reject("D:\\p\\src\\a.js".replace("\\", __import__("os").sep), blocked)


# ---------------------------------------------------------------------- scheduling
class FakeGate:
    def __init__(self) -> None:
        self.on_ac = True
        self.ok = True
        self.why = ""
        self.updates = 0

    def update(self) -> None:
        self.updates += 1

    def ready(self, allow_battery: bool = False) -> tuple[bool, str]:
        return self.ok, self.why


class FakeHandle:
    def __init__(self) -> None:
        self.code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[bool] = []
        self.handles: list[FakeHandle] = []

    def start(self, reconcile: bool) -> FakeHandle:
        self.started.append(reconcile)
        handle = FakeHandle()
        self.handles.append(handle)
        return handle


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def parts(env: Env, clock: Clock) -> tuple[Watcher, FakeGate, FakeLauncher, list[int]]:
    gate, launcher, unloads = FakeGate(), FakeLauncher(), []
    watcher_ = Watcher(
        env.settings,
        env.state,
        gate,
        launcher,
        lambda: unloads.append(1),
        projects=env.projects,
        scope=env.scope,
        clock=clock,
        roots=[str(env.root)],
    )
    env.state.set_meta("last_reconcile", str(clock.now))  # no reconcile due unless a test says so
    return watcher_, gate, launcher, unloads


class TestScheduling:
    def test_spawns_a_worker_when_ready_and_work_is_queued(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, gate, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        assert launcher.started == [False]
        assert gate.updates == 1

    def test_nothing_queued_means_no_worker(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]]
    ) -> None:
        w, _, launcher, _ = parts
        w.tick()
        assert launcher.started == []

    def test_debounced_items_do_not_count_as_due(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, _, launcher, _ = parts
        env.state.enqueue("a", delay=DEBOUNCE * 100)
        w.tick()
        assert launcher.started == []

    def test_gate_not_ready_blocks_and_logs_reason_changes(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, gate, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        gate.ok, gate.why = False, "battery, queue only"
        w.tick()
        assert launcher.started == []
        assert w.last_reason == "battery, queue only"
        gate.ok, gate.why = True, ""
        w.tick()
        assert launcher.started == [False]
        assert w.last_reason == ""

    def test_reconcile_runs_when_due_even_with_an_empty_queue(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, _, launcher, _ = parts
        clock.now += 7 * 3600
        w.tick()
        assert launcher.started == [True]
        assert w.next_spawn > clock.now  # does not immediately re-trigger

    def test_running_worker_blocks_a_second_spawn_and_exit_clears_it(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, _, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        w.tick()
        assert launcher.started == [False]
        env.state.done([("a", env.state.claim(1)[0].seq)])  # the worker finished something
        env.state.enqueue("b", delay=0)
        launcher.handles[0].code = 0
        w.tick()  # exit noticed; more work is queued, so a fresh worker starts
        assert launcher.started == [False, False]

    def test_failed_worker_backs_off(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, _, launcher, unloads = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        launcher.handles[0].code = 2
        w.tick()
        assert unloads == [1]  # a crashed worker never unloaded the model itself
        assert launcher.started == [False]
        clock.now += 601
        w.tick()
        assert launcher.started == [False, False]

    def test_unplugging_terminates_a_stuck_worker_after_the_grace_period(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, gate, launcher, unloads = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        gate.on_ac = False
        w.tick()
        assert not launcher.handles[0].terminated
        clock.now += 21
        w.tick()
        assert launcher.handles[0].terminated
        assert unloads == [1]

    def test_a_worker_that_makes_no_progress_is_not_restarted_at_once(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, _, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        launcher.handles[0].code = 0  # exited cleanly, but the queue did not shrink
        w.tick()
        w.tick()
        assert launcher.started == [False]
        clock.now += 121
        w.tick()
        assert launcher.started == [False, False]

    def test_a_worker_that_drains_the_queue_leaves_no_backoff(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, _, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        env.state.done([("a", env.state.claim(1)[0].seq)])
        env.state.enqueue("b", delay=0)  # new work arrives right after
        launcher.handles[0].code = 0
        w.tick()
        assert launcher.started == [False, False]

    def test_a_clean_exit_does_not_ask_for_an_unload(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, _, launcher, unloads = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        launcher.handles[0].code = 0
        w.tick()
        assert unloads == []

    def test_a_stop_request_ends_the_loop_without_scheduling(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env
    ) -> None:
        w, _, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        request_stop(env.settings.storage.data_dir)
        w.tick()
        assert launcher.started == []
        assert w._stop.is_set()

    def test_shutdown_lets_a_worker_exit_by_itself_before_killing_it(
        self,
        parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]],
        env: Env,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        w, _, launcher, unloads = parts
        monkeypatch.setattr(watcher, "_WORKER_STOP_GRACE_SECONDS", 2.0)
        env.state.enqueue("a", delay=0)
        w.tick()
        handle = launcher.handles[0]
        polls = iter([None, None, 0])
        handle.poll = lambda: next(polls, 0)  # type: ignore[method-assign]
        w._stop_worker()
        assert not handle.terminated
        assert unloads == [1]

    def test_replugging_resets_the_grace_timer(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, gate, launcher, _ = parts
        env.state.enqueue("a", delay=0)
        w.tick()
        gate.on_ac = False
        w.tick()
        clock.now += 15
        gate.on_ac = True
        w.tick()
        assert w.unplugged_at is None
        gate.on_ac = False
        clock.now += 15
        w.tick()
        assert not launcher.handles[0].terminated

    def test_reconcile_due_uses_the_last_run(
        self, parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]], env: Env, clock: Clock
    ) -> None:
        w, *_ = parts
        assert not w.reconcile_due()
        env.state.set_meta("last_reconcile", "0")
        assert w.reconcile_due()


class TestLifecycle:
    def test_run_watches_files_ticks_and_cleans_up(
        self,
        env: Env,
        parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import threading

        w, _, launcher, unloads = parts
        env.settings = env.settings.model_copy(
            update={"idle": env.settings.idle.model_copy(update={"file_debounce_seconds": 0})}
        )
        w.settings = env.settings
        monkeypatch.setattr(watcher, "TICK_SECONDS", 0.05)
        monkeypatch.setattr(watcher, "_WORKER_STOP_GRACE_SECONDS", 0.2)
        thread = threading.Thread(target=w.run, daemon=True)
        thread.start()
        path = write(env, "live.py", "value = 1\n")
        deadline = time.time() + 10
        while time.time() < deadline and not launcher.started:
            time.sleep(0.05)
        w.stop()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert path in queued(env)
        assert launcher.started
        assert unloads == [1]  # the still-running worker was terminated and the model unloaded
        assert launcher.handles[0].terminated

    def test_a_failing_tick_does_not_stop_the_loop(
        self,
        parts: tuple[Watcher, FakeGate, FakeLauncher, list[int]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import threading

        w, *_ = parts
        calls: list[int] = []

        def failing_tick() -> None:
            calls.append(1)
            raise RuntimeError("boom")

        monkeypatch.setattr(w, "tick", failing_tick)
        monkeypatch.setattr(watcher, "TICK_SECONDS", 0.02)
        thread = threading.Thread(target=w.run, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while len(calls) < 3 and time.time() < deadline:
            time.sleep(0.02)
        w.stop()
        thread.join(timeout=5)
        assert len(calls) >= 3

    def test_missing_roots_are_skipped(self, env: Env, clock: Clock) -> None:
        w = Watcher(
            env.settings,
            env.state,
            FakeGate(),
            FakeLauncher(),
            lambda: None,
            projects=env.projects,
            scope=env.scope,
            clock=clock,
            roots=[str(env.root / "nope")],
        )
        assert w.roots == []


class TestHelpers:
    def test_status_summarises_the_index(self, env: Env) -> None:
        env.state.manifest_set("a", 1, 1, "h")
        env.state.enqueue("b", delay=0)
        text = watcher.status(env.settings)
        assert "indexed files   : 1" in text
        assert "queue           : 1 total, 1 due" in text
        assert "last reconcile  : never" in text

    @staticmethod
    def fake_ollama(
        monkeypatch: pytest.MonkeyPatch, resident: list[dict[str, object]]
    ) -> list[tuple[str, bytes | None]]:
        sent: list[tuple[str, bytes | None]] = []

        class Response:
            def __init__(self, payload: object) -> None:
                self._payload = payload

            def read(self) -> bytes:
                return json.dumps(self._payload).encode()

        def fake_urlopen(request: object, timeout: float) -> Response:
            url = request.full_url  # type: ignore[attr-defined]
            sent.append((url, request.data))  # type: ignore[attr-defined]
            return Response({"models": resident} if url.endswith("/api/ps") else {})

        monkeypatch.setattr(ollama_http.urllib.request, "urlopen", fake_urlopen)
        return sent

    def test_unload_model_posts_keep_alive_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = self.fake_ollama(monkeypatch, [{"model": "m:latest", "size_vram": 5}])
        ollama_http.unload_model("http://host:1/", "m")
        assert [url for url, _ in sent] == ["http://host:1/api/ps", "http://host:1/api/embed"]
        body = json.loads(sent[1][1] or b"")
        assert body["keep_alive"] == 0
        assert "options" not in body

    def test_unload_model_keeps_a_cpu_runner_on_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sent = self.fake_ollama(monkeypatch, [{"model": "m", "size_vram": 0}])
        ollama_http.unload_model("http://host:1", "m")
        assert json.loads(sent[1][1] or b"")["options"] == {"num_gpu": 0}

    def test_unload_model_does_not_load_what_is_not_resident(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent = self.fake_ollama(monkeypatch, [{"model": "other", "size_vram": 5}])
        ollama_http.unload_model("http://host:1", "m")
        assert [url for url, _ in sent] == ["http://host:1/api/ps"]

    def test_unload_model_swallows_connection_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*_a: object, **_k: object) -> None:
            raise ConnectionRefusedError

        monkeypatch.setattr(ollama_http.urllib.request, "urlopen", refuse)
        ollama_http.unload_model("http://host:1", "m")

    def test_build_watcher_wires_real_collaborators(self, env: Env) -> None:
        w = watcher.build_watcher(env.settings, env.state)
        assert isinstance(w.launcher, watcher.SubprocessLauncher)
        assert w.gate is not None

    def test_subprocess_launcher_builds_the_worker_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_popen(command: list[str], **kwargs: object) -> str:
            captured["command"] = command
            captured["flags"] = kwargs["creationflags"]
            return "handle"

        monkeypatch.setattr(watcher.subprocess, "Popen", fake_popen)
        launcher = watcher.SubprocessLauncher(tmp_path / "logs")
        assert launcher.start(reconcile=True) == "handle"
        assert captured["command"][-4:] == ["-m", "vector_embed", "worker", "--reconcile"]  # type: ignore[index]
        assert captured["flags"] == 0x00004000 | 0x08000000
        launcher.start(reconcile=False)
        assert captured["command"][-3:] == ["-m", "vector_embed", "worker"]  # type: ignore[index]

    def test_main_status(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(watcher, "load_settings", lambda: env.settings)
        assert watcher.main(["--status"]) == 0
        assert "indexed files" in capsys.readouterr().out

    def test_main_exits_when_already_running(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from vector_embed.core.process import single_instance

        monkeypatch.setattr(watcher, "load_settings", lambda: env.settings)
        monkeypatch.setattr(watcher, "configure_logging", lambda *_a, **_k: None)
        with single_instance("watcher", env.data_dir):
            assert watcher.main([]) == 0

    def test_main_runs_the_watcher_until_stopped(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran: list[str] = []

        class Stub:
            def run(self) -> None:
                ran.append("run")

            def stop(self) -> None:
                ran.append("stop")

        monkeypatch.setattr(watcher, "load_settings", lambda: env.settings)
        monkeypatch.setattr(watcher, "configure_logging", lambda *_a, **_k: None)
        monkeypatch.setattr(watcher, "build_watcher", lambda _s, _st: Stub())
        monkeypatch.setattr(watcher.signal, "signal", lambda *_a: None)
        factory: Callable[..., object] = watcher.main
        assert factory([]) == 0
        assert ran == ["run"]
