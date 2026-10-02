from pathlib import Path

from tests.core.conftest import Env

from vector_embed.core import doctor
from vector_embed.core.extractors.base import NullOcr
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.providers.base import CAP_COMPLETION, CAP_EMBEDDING, ModelInfo

GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)


class Provider:
    name = "ollama"

    def __init__(self, models: list[ModelInfo]) -> None:
        self.models = models

    def list_models(self) -> list[ModelInfo]:
        return self.models


class Ocr:
    def available(self) -> bool:
        return True

    def ocr_image(self, image: object) -> str:
        return ""


def registry(models: list[ModelInfo]) -> ModelRegistry:
    return ModelRegistry(load_catalog(), [Provider(models)], hardware_probe=lambda: GPU)


def names(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.name: c for c in checks}


def test_healthy_machine(env: Env) -> None:
    embed = ModelInfo(
        "qwen3-embedding:0.6b", "ollama", capabilities=frozenset({CAP_EMBEDDING}), size_bytes=1
    )
    chat = ModelInfo("qwen3.5:9b", "ollama", capabilities=frozenset({CAP_COMPLETION}), size_bytes=1)
    checks = names(
        doctor.run_doctor(
            env.settings,
            registry([embed, chat]),
            GPU,
            Ocr(),
            state=env.state,
            which=lambda _n: "git",
        )
    )
    assert all(c.ok for c in checks.values())
    assert checks["chat model"].detail == "qwen3.5:9b"
    assert "files indexed" in checks["index"].detail
    assert doctor.format_checks(list(checks.values())).count("[ok]") == len(checks)


def test_failures_explain_the_fix(env: Env) -> None:
    cpu_only = Hardware(None, 0, 0, 16000, 8000, 4, False)
    checks = names(
        doctor.run_doctor(env.settings, registry([]), cpu_only, NullOcr(), which=lambda _n: None)
    )
    assert not checks["ollama"].ok
    assert "ollama pull qwen3-embedding:0.6b" in checks["embedding model"].detail
    assert "ollama pull qwen3.5:9b" in checks["chat model"].detail
    assert not checks["gpu"].ok
    assert not checks["windows ocr"].ok
    assert not checks["git"].ok
    assert checks["power"].detail.startswith("battery")
    assert "index" not in checks
    assert "[FAIL]" in doctor.format_checks(list(checks.values()))


def test_unwritable_data_dir(env: Env, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    check = doctor._data_dir_check(blocker / "sub")
    assert not check.ok
    assert "not writable" in check.detail


def test_old_python_is_flagged(monkeypatch: object) -> None:
    import sys

    class Fake:
        pass

    original = sys.version_info
    try:
        sys.version_info = (3, 9, 0, "final", 0)  # type: ignore[assignment]
        check = doctor._python_check()
    finally:
        sys.version_info = original
    assert not check.ok
    assert "3.11" in check.detail
