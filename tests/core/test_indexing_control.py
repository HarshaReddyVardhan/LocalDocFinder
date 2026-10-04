from pathlib import Path

import pytest

from localdoc_finder.core.indexing_control import IndexingControl, IndexingStatus
from localdoc_finder.core.process import single_instance
from localdoc_finder.core.store.sqlite import StateDb


class FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def start(self, reconcile: bool, now: bool = False) -> object:
        self.calls.append((reconcile, now))
        return object()


@pytest.fixture
def state(tmp_path: Path) -> StateDb:
    with StateDb(tmp_path) as db:
        yield db  # type: ignore[misc]


def control(state: StateDb, tmp_path: Path, launcher: FakeLauncher, on_ac: bool = True):  # type: ignore[no-untyped-def]
    return IndexingControl(state, tmp_path, launcher, lambda: on_ac)


def test_start_runs_now_with_a_scan_and_clears_a_pause(state: StateDb, tmp_path: Path) -> None:
    launcher = FakeLauncher()
    box = control(state, tmp_path, launcher)
    box.pause()
    assert box.is_paused()
    result = box.start()
    assert result.started
    assert "where it left off" in result.message
    assert not box.is_paused()
    assert launcher.calls == [(True, True)]  # scan for files, and do not wait for idle


def test_start_never_runs_on_battery(state: StateDb, tmp_path: Path) -> None:
    launcher = FakeLauncher()
    result = control(state, tmp_path, launcher, on_ac=False).start()
    assert not result.started
    assert "plugged in" in result.message
    assert launcher.calls == []


def test_start_when_already_running_does_nothing(state: StateDb, tmp_path: Path) -> None:
    launcher = FakeLauncher()
    box = control(state, tmp_path, launcher)
    with single_instance("worker", tmp_path) as acquired:  # a worker holds its lock
        assert acquired
        assert box.is_running()
        assert "already running" in box.start().message
        box.pause()
        assert "still stopping" in box.start().message
        assert box.is_paused()  # not cleared while the worker is winding down
    assert launcher.calls == []
    assert not box.is_running()


def test_status_counts_and_summaries(state: StateDb, tmp_path: Path) -> None:
    state.manifest_set("D:/a.txt", 1, 1, "h")
    state.enqueue("D:/b.txt", delay=0)
    status = control(state, tmp_path, FakeLauncher()).status()
    assert (status.indexed, status.waiting, status.running, status.paused) == (1, 1, False, False)
    assert status.fraction == 0.5
    assert "1 files indexed, 1 waiting" in status.summary
    assert "waiting for the PC" in status.summary
    assert "paused" in IndexingStatus(1, 1, False, True).summary
    assert "finishing" in IndexingStatus(1, 1, True, True).summary
    assert IndexingStatus(5, 0, True, False).summary.startswith("Indexing: indexing")
    assert IndexingStatus(0, 0, False, False).fraction == 1.0
