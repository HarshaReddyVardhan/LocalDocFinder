import sys
import time
from pathlib import Path

import pytest

from localdoc_finder.core.process import ENTRY_POINTS, self_command, single_instance


def test_second_holder_is_refused_until_released(tmp_path: Path) -> None:
    with single_instance("worker", tmp_path) as first:
        assert first is True
        with single_instance("worker", tmp_path) as second:
            assert second is False
    with single_instance("worker", tmp_path) as again:
        assert again is True


def test_locks_are_per_name(tmp_path: Path) -> None:
    with single_instance("a", tmp_path) as a, single_instance("b", tmp_path) as b:
        assert a is True
        assert b is True


def test_creates_the_data_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    with single_instance("x", target) as got:
        assert got is True
    assert target.is_dir()


def test_self_command_from_source_runs_the_package() -> None:
    command = self_command("watcher")
    assert command[0] == sys.executable
    assert command[1:] == ["-m", "localdoc_finder", "watcher"]


def test_self_command_frozen_runs_the_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", r"C:\Apps\LocalDocFinder\LocalDocFinder.exe")
    assert self_command("worker") == [r"C:\Apps\LocalDocFinder\LocalDocFinder.exe", "worker"]
    assert self_command("app", windowless=True)[1:] == ["app"]


def test_windowless_uses_pythonw_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    monkeypatch.setattr(sys, "executable", str(python))
    assert self_command("app", windowless=True)[0] == str(python)  # no pythonw next to it
    (tmp_path / "pythonw.exe").write_bytes(b"")
    assert self_command("app", windowless=True)[0] == str(tmp_path / "pythonw.exe")
    assert self_command("app")[0] == str(python)


def test_unknown_entry_point_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown entry point"):
        self_command("bogus")
    assert set(ENTRY_POINTS) == {"app", "watcher", "worker", "setup"}


def test_stop_request_round_trip_and_expiry(tmp_path: Path) -> None:
    from localdoc_finder.core.process import clear_stop_request, request_stop, stop_requested

    assert not stop_requested(tmp_path)
    request_stop(tmp_path)
    assert stop_requested(tmp_path)
    assert not stop_requested(tmp_path, now=time.time() + 3600)  # a stale flag is ignored
    clear_stop_request(tmp_path)
    assert not stop_requested(tmp_path)
