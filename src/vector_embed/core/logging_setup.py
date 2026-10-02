"""Structured logging: JSON lines in log files, readable text on the console."""

import json
import logging
import logging.handlers
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

LOG_MAX_BYTES = 2 * 1024 * 1024  # per log file; the always-on processes must not fill the disk
LOG_BACKUPS = 3

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line; ``extra={...}`` fields become top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(name: str, log_dir: Path | None, level: str = "INFO") -> None:
    """Configure the root logger for a process (``worker``, ``watcher``, ``app``, ``cli``).

    Safe to call more than once: earlier handlers installed by this function are replaced.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_vector_embed", False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    handlers: list[logging.Handler] = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        handlers.append(file_handler)
    if sys.stderr is not None:  # pythonw has no console
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        handlers.append(console)
    for handler in handlers:
        handler._vector_embed = True  # type: ignore[attr-defined]  # marker for reconfiguration
        root.addHandler(handler)


def rotate_if_large(path: Path, max_bytes: int = LOG_MAX_BYTES, backups: int = LOG_BACKUPS) -> None:
    """Shift ``path`` to ``path.1`` (and older ones along) once it is bigger than ``max_bytes``.

    For files another program appends to (a child process's redirected output), where a
    logging handler cannot do the rotating.
    """
    try:
        if path.stat().st_size <= max_bytes:
            return
    except OSError:
        return  # nothing there yet
    try:
        for index in range(backups, 0, -1):
            older = path.with_name(f"{path.name}.{index}")
            newer = path if index == 1 else path.with_name(f"{path.name}.{index - 1}")
            if newer.exists():
                older.unlink(missing_ok=True)
                newer.rename(older)
    except OSError:
        logging.getLogger(__name__).warning("could not rotate %s", path, exc_info=True)


def install_excepthooks() -> None:
    """Send uncaught exceptions to the log; the windowed exe has no stderr to show them on."""
    logger = logging.getLogger("uncaught")

    def on_main_thread(
        exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))

    def on_other_thread(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        logger.critical(
            "uncaught exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = on_main_thread
    threading.excepthook = on_other_thread
