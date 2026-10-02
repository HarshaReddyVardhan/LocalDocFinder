"""``ve doctor``: quick health checks with an actionable message for each failure."""

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vector_embed.core.extractors.base import OcrEngine
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.settings import Settings
from vector_embed.core.store.sqlite import StateDb

MIN_PYTHON = (3, 11)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def run_doctor(
    settings: Settings,
    registry: ModelRegistry,
    hardware: Hardware,
    ocr: OcrEngine,
    *,
    state: StateDb | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Check]:
    checks = [_python_check(), _data_dir_check(settings.storage.data_dir)]
    checks.append(_ollama_check(registry))
    checks.append(_embed_model_check(settings, registry))
    resolutions = registry.resolve_all(hardware)
    chat = resolutions["chat"]
    checks.append(
        Check(
            "chat model",
            chat.model is not None,
            chat.model or f"none usable; try: ollama pull {_first_chat_choice(registry)}",
        )
    )
    checks.append(
        Check(
            "gpu",
            hardware.has_gpu,
            f"{hardware.gpu_name}, {hardware.vram_free_mb} MB free"
            if hardware.has_gpu
            else "no NVIDIA GPU: search works, chat will be slow",
        )
    )
    checks.append(
        Check(
            "windows ocr",
            ocr.available(),
            "available" if ocr.available() else "unavailable: images and scanned PDFs get no text",
        )
    )
    checks.append(
        Check(
            "git",
            which("git") is not None,
            "found"
            if which("git")
            else "not on PATH: .gitignore handling falls back to a slower scan",
        )
    )
    checks.append(Check("power", True, "AC" if hardware.on_ac else "battery (indexing paused)"))
    if state is not None:
        checks.append(
            Check(
                "index",
                True,
                f"{state.manifest_count()} files indexed, {state.queue_size()} queued",
            )
        )
    return checks


def format_checks(checks: list[Check]) -> str:
    return "\n".join(f"[{'ok' if c.ok else 'FAIL'}] {c.name}: {c.detail}" for c in checks)


def _python_check() -> Check:
    version = sys.version_info[:2]
    return Check(
        "python",
        version >= MIN_PYTHON,
        f"{version[0]}.{version[1]}" + ("" if version >= MIN_PYTHON else " (needs 3.11+)"),
    )


def _data_dir_check(data_dir: Path) -> Check:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("data dir", False, f"{data_dir} is not writable: {exc}")
    return Check("data dir", True, str(data_dir))


def _ollama_check(registry: ModelRegistry) -> Check:
    models = registry.refresh()
    if models:
        return Check("ollama", True, f"{len(models)} models installed")
    return Check("ollama", False, "no models found; start Ollama (`ollama serve`) and pull models")


def _embed_model_check(settings: Settings, registry: ModelRegistry) -> Check:
    wanted = settings.embedding.model
    installed = {m.name.removesuffix(":latest") for m in registry.installed}
    if wanted.removesuffix(":latest") in installed:
        return Check("embedding model", True, wanted)
    return Check("embedding model", False, f"{wanted} is missing; run: ollama pull {wanted}")


def _first_chat_choice(registry: ModelRegistry) -> str:
    preferences = registry.preferences("chat")
    return preferences[0] if preferences else "qwen3.5:9b"
