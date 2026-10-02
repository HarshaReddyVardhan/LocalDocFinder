import subprocess
from pathlib import Path

import psutil
import pytest

from vector_embed.core import lifecycle
from vector_embed.core.autostart import Autostart
from vector_embed.core.store.sqlite import STATE_FILENAME


class FakeProcess:
    def __init__(self, pid: int, name: str, *, stubborn: bool = False, fail: bool = False) -> None:
        self._pid = pid
        self._name = name
        self.stubborn = stubborn
        self.fail = fail
        self.terminated = False
        self.killed = False

    @property
    def pid(self) -> int:
        return self._pid

    def name(self) -> str:
        if self.fail:
            raise psutil.NoSuchProcess(self._pid)
        return self._name

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def stop(procs: list[FakeProcess], current: int = 1) -> int:
    def alive(
        processes: list[lifecycle.ProcessLike], _timeout: float
    ) -> list[lifecycle.ProcessLike]:
        return [p for p in processes if getattr(p, "stubborn", False)]

    return lifecycle.stop_other_instances(current_pid=current, lister=lambda: procs, wait=alive)


def test_stops_other_copies_but_not_itself_or_strangers() -> None:
    me = FakeProcess(1, "VectorEmbed.exe")
    watcher = FakeProcess(2, "vectorembed.EXE")
    other = FakeProcess(3, "notepad.exe")
    assert stop([me, watcher, other]) == 1
    assert watcher.terminated
    assert not (me.terminated or other.terminated)


def test_stops_the_cli_too() -> None:
    cli = FakeProcess(4, "ve.exe")
    assert stop([cli]) == 1
    assert cli.terminated


def test_stubborn_processes_are_killed() -> None:
    stubborn = FakeProcess(2, "VectorEmbed.exe", stubborn=True)
    assert stop([stubborn]) == 1
    assert stubborn.killed


def test_processes_that_vanish_are_skipped() -> None:
    assert stop([FakeProcess(2, "VectorEmbed.exe", fail=True)]) == 0


def test_terminate_errors_are_tolerated() -> None:
    class Unkillable(FakeProcess):
        def terminate(self) -> None:
            raise psutil.AccessDenied(self.pid)

        def kill(self) -> None:
            raise psutil.AccessDenied(self.pid)

    assert stop([Unkillable(2, "VectorEmbed.exe", stubborn=True)]) == 1


def test_wait_for_exit_returns_the_survivors() -> None:
    assert lifecycle.wait_for_exit([], 0.01) == []


def test_default_lister_lists_this_process() -> None:
    assert any(p.pid == psutil.Process().pid for p in lifecycle.list_processes())


def test_start_watcher_spawns_it_detached() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    lifecycle.start_watcher(lambda argv, **kw: calls.append((argv, kw)))
    argv, kwargs = calls[0]
    assert argv[-1] == "watcher"
    assert kwargs["creationflags"] == 0x00000008 | 0x08000000
    assert kwargs["stdout"] is subprocess.DEVNULL


def test_start_watcher_survives_a_launch_failure() -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("blocked")

    lifecycle.start_watcher(boom)


def test_data_deletion_waits_for_this_process_then_removes_the_folder(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    target = tmp_path / "it's data"
    target.mkdir()
    (target / STATE_FILENAME).write_text("")
    lifecycle.schedule_data_deletion(target, pid=4242, spawn=lambda argv, **_kw: calls.append(argv))
    script = calls[0][-1]
    assert "Wait-Process -Id 4242" in script
    assert "Remove-Item -LiteralPath" in script
    assert "it''s data" in script  # single quotes are doubled for PowerShell
    assert "-Recurse -Force" in script
    assert "-ErrorAction Stop" in script  # a failed delete is retried, not hidden


def test_data_deletion_refuses_a_folder_without_the_state_database(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    (tmp_path / "notes.txt").write_text("my own files")
    with pytest.raises(lifecycle.DataDeletionRefusedError):
        lifecycle.schedule_data_deletion(tmp_path, spawn=lambda argv, **_kw: calls.append(argv))
    assert calls == []


class RecordingAutostart(Autostart):
    def __init__(self) -> None:
        super().__init__(lambda _script: 0)
        self.calls: list[str] = []

    def register(self) -> None:
        self.calls.append("register")

    def unregister(self) -> None:
        self.calls.append("unregister")


def test_after_install_registers_startup_only_when_enabled() -> None:
    tasks = RecordingAutostart()
    lifecycle.after_install(tasks, lambda: True)
    lifecycle.after_install(tasks, lambda: False)
    assert tasks.calls == ["register"]


def test_hooks_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(RecordingAutostart):
        def register(self) -> None:
            raise RuntimeError("scheduler unavailable")

        def unregister(self) -> None:
            raise RuntimeError("scheduler unavailable")

    monkeypatch.setattr(lifecycle, "start_watcher", lambda: (_ for _ in ()).throw(OSError("no")))
    monkeypatch.setattr(lifecycle, "stop_other_instances", lambda: (_ for _ in ()).throw(OSError()))
    lifecycle.after_install(Broken(), lambda: True)
    lifecycle.after_update(Broken(), lambda: True)
    lifecycle.before_update()
    lifecycle.before_uninstall(Broken())


def test_update_and_uninstall_chores_run_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(lifecycle, "start_watcher", lambda: order.append("watcher"))
    monkeypatch.setattr(lifecycle, "stop_other_instances", lambda: order.append("stop") or 0)
    tasks = RecordingAutostart()
    lifecycle.before_update()
    lifecycle.after_update(tasks, lambda: True)
    lifecycle.before_uninstall(tasks)
    assert order == ["stop", "watcher", "stop"]
    assert tasks.calls == ["register", "unregister"]


class FakeVelopackApp:
    def __init__(self) -> None:
        self.hooks: dict[str, object] = {}
        self.ran = False

    def _hook(self, name: str, callback: object) -> "FakeVelopackApp":
        self.hooks[name] = callback
        return self

    def on_after_install_fast_callback(self, callback: object) -> "FakeVelopackApp":
        return self._hook("install", callback)

    def on_after_update_fast_callback(self, callback: object) -> "FakeVelopackApp":
        return self._hook("update", callback)

    def on_before_update_fast_callback(self, callback: object) -> "FakeVelopackApp":
        return self._hook("before_update", callback)

    def on_before_uninstall_fast_callback(self, callback: object) -> "FakeVelopackApp":
        return self._hook("uninstall", callback)

    def run(self) -> None:
        self.ran = True


def test_startup_hooks_are_wired_to_velopack(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(lifecycle, "start_watcher", lambda: order.append("watcher"))
    monkeypatch.setattr(lifecycle, "stop_other_instances", lambda: order.append("stop") or 0)
    app = FakeVelopackApp()
    tasks = RecordingAutostart()
    lifecycle.run_startup_hooks(lambda: app, autostart=tasks)  # type: ignore[arg-type,return-value]
    assert app.ran
    assert set(app.hooks) == {"install", "update", "before_update", "uninstall"}
    for name in ("install", "before_update", "update", "uninstall"):
        app.hooks[name]("1.2.3")  # type: ignore[operator]  # Velopack passes the version
    assert tasks.calls == ["register", "register", "unregister"]
    assert order == ["stop", "watcher", "stop"]


def test_startup_hooks_default_to_the_real_velopack_app() -> None:
    lifecycle.run_startup_hooks(autostart=RecordingAutostart())  # no install flags: returns
