import json
import logging
import sys
import threading
from pathlib import Path

import pytest

from localdoc_finder.core import logging_setup
from localdoc_finder.core.logging_setup import (
    JsonFormatter,
    configure_logging,
    install_excepthooks,
    rotate_if_large,
)


@pytest.fixture(autouse=True)
def restore_root_logger() -> object:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_json_formatter_includes_extra_fields_and_exceptions() -> None:
    logger = logging.getLogger("t")
    try:
        raise ValueError("bad")
    except ValueError:
        record = logger.makeRecord(
            "t", logging.ERROR, __file__, 1, "failed %s", ("x",), None, extra={"path": "a.py"}
        )
        record.exc_info = __import__("sys").exc_info()
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "failed x"
    assert data["level"] == "ERROR"
    assert data["path"] == "a.py"
    assert "ValueError" in data["exception"]
    assert data["ts"].endswith("+00:00")


def test_configure_writes_json_lines_to_the_log_file(tmp_path: Path) -> None:
    configure_logging("worker", tmp_path / "logs", "INFO")
    logging.getLogger("demo").info("hello", extra={"files": 3})
    logging.getLogger("demo").debug("hidden")
    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = (tmp_path / "logs" / "worker.log").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert (record["message"], record["files"]) == ("hello", 3)
    assert all("hidden" not in line for line in lines)


def test_reconfiguring_replaces_handlers(tmp_path: Path) -> None:
    root = logging.getLogger()
    configure_logging("a", tmp_path, "INFO")
    first = len(root.handlers)
    configure_logging("b", tmp_path, "INFO")
    assert len(root.handlers) == first
    assert (tmp_path / "b.log").exists()


def test_console_only_when_no_log_dir(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("cli", None, "INFO")
    logging.getLogger("x").warning("to console")
    assert "to console" in capsys.readouterr().err


def test_uncaught_exceptions_reach_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    install_excepthooks()
    with caplog.at_level(logging.CRITICAL, logger="uncaught"):
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())  # type: ignore[arg-type]
        worker = threading.Thread(target=lambda: 1 / 0, name="job")
        worker.start()
        worker.join()
    messages = [record.getMessage() for record in caplog.records]
    assert "uncaught exception" in messages
    assert "uncaught exception in thread job" in messages


def test_log_files_rotate_instead_of_growing_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logging_setup, "LOG_MAX_BYTES", 2000)
    monkeypatch.setattr(logging_setup, "LOG_BACKUPS", 2)
    configure_logging("rot", tmp_path, "INFO")
    logger = logging.getLogger("rot-test")
    for i in range(200):
        logger.info("filler line number %d with some padding to take space", i)
    names = sorted(p.name for p in tmp_path.glob("rot.log*"))
    assert names == ["rot.log", "rot.log.1", "rot.log.2"]  # a bounded set of files
    assert all(p.stat().st_size < 3000 for p in tmp_path.glob("rot.log*"))


class TestRotateIfLarge:
    def test_a_big_file_moves_aside_and_older_ones_shift(self, tmp_path: Path) -> None:
        log = tmp_path / "worker.out.log"
        log.write_text("x" * 100)
        (tmp_path / "worker.out.log.1").write_text("older")
        rotate_if_large(log, max_bytes=50, backups=2)
        assert not log.exists()  # the writer starts a fresh file
        assert (tmp_path / "worker.out.log.1").read_text() == "x" * 100
        assert (tmp_path / "worker.out.log.2").read_text() == "older"

    def test_the_oldest_backup_is_dropped(self, tmp_path: Path) -> None:
        log = tmp_path / "w.log"
        for suffix, text in (("", "now"), (".1", "one"), (".2", "two")):
            (tmp_path / f"w.log{suffix}").write_text(text * 40)
        rotate_if_large(log, max_bytes=10, backups=2)
        assert (tmp_path / "w.log.2").read_text() == "one" * 40  # "two" is gone
        assert not (tmp_path / "w.log.3").exists()

    def test_a_small_or_missing_file_is_left_alone(self, tmp_path: Path) -> None:
        small = tmp_path / "small.log"
        small.write_text("tiny")
        rotate_if_large(small, max_bytes=50)
        assert small.read_text() == "tiny"
        rotate_if_large(tmp_path / "missing.log")  # no error
