from types import SimpleNamespace

import pytest

from localdoc_finder.core.models import hardware as hwmod


def test_probe_gpu_returns_none_without_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    assert hwmod.probe_gpu() is None


def test_probe_gpu_reads_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    calls: list[str] = []
    fake = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda _i: object(),
        nvmlDeviceGetName=lambda _h: b"RTX 2070",
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(
            total=8192 * 1024**2, free=7000 * 1024**2
        ),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert hwmod.probe_gpu() == ("RTX 2070", 8192, 7000)
    assert calls == ["init", "shutdown"]


def test_probe_gpu_accepts_str_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _i: object(),
        nvmlDeviceGetName=lambda _h: "GPU",
        nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(total=1024**2, free=1024**2),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert hwmod.probe_gpu() == ("GPU", 1, 1)


@pytest.mark.parametrize(
    ("battery", "expected"),
    [
        (None, True),
        (SimpleNamespace(power_plugged=None), False),  # unknown: fail closed
        (SimpleNamespace(power_plugged=True), True),
        (SimpleNamespace(power_plugged=False), False),
    ],
)
def test_on_ac_power(
    monkeypatch: pytest.MonkeyPatch, battery: SimpleNamespace | None, expected: bool
) -> None:
    monkeypatch.setattr(hwmod.psutil, "sensors_battery", lambda: battery)
    assert hwmod.on_ac_power() is expected


def test_on_ac_power_fails_closed_on_psutil_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise OSError

    monkeypatch.setattr(hwmod.psutil, "sensors_battery", boom)
    assert hwmod.on_ac_power() is False  # an unreadable probe must never allow battery indexing


def test_probe_hardware_combines_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hwmod, "probe_gpu", lambda: ("G", 8000, 6000))
    monkeypatch.setattr(hwmod, "on_ac_power", lambda: False)
    result = hwmod.probe_hardware()
    assert result.has_gpu
    assert (result.vram_total_mb, result.vram_free_mb) == (8000, 6000)
    assert result.on_ac is False
    assert result.ram_total_mb > 0
    assert result.cpu_count >= 1


def test_probe_hardware_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hwmod, "probe_gpu", lambda: None)
    result = hwmod.probe_hardware()
    assert not result.has_gpu
    assert result.vram_free_mb == 0
