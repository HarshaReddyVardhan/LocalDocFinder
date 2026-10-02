"""Structured logging: JSON lines in log files, readable text on the console."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
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
