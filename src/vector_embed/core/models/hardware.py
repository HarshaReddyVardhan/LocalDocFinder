"""Hardware probe: GPU/VRAM, RAM, CPU and power source. Every probe degrades gracefully."""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


@dataclass(frozen=True)
class Hardware:
    gpu_name: str | None
    vram_total_mb: int
    vram_free_mb: int
    ram_total_mb: int
    ram_free_mb: int
    cpu_count: int
    on_ac: bool

    @property
    def has_gpu(self) -> bool:
        return self.gpu_name is not None


def probe_gpu() -> tuple[str, int, int] | None:
    """``(name, total_mb, free_mb)`` of the first NVIDIA GPU, or ``None`` without NVML."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return name, int(memory.total) // _MB, int(memory.free) // _MB
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # NVML missing, no driver or no device: all mean "no usable GPU"
        logger.debug("hardware: no NVIDIA GPU available", exc_info=True)
        return None


def on_ac_power() -> bool:
    """True if plugged in. No battery (a desktop) counts as plugged in."""
    try:
        battery = psutil.sensors_battery()
    except Exception:  # psutil can raise on odd platforms; assume plugged in
        return True
    if battery is None or battery.power_plugged is None:
        return True
    return bool(battery.power_plugged)


def probe_hardware() -> Hardware:
    gpu = probe_gpu()
    memory = psutil.virtual_memory()
    return Hardware(
        gpu_name=gpu[0] if gpu else None,
        vram_total_mb=gpu[1] if gpu else 0,
        vram_free_mb=gpu[2] if gpu else 0,
        ram_total_mb=int(memory.total) // _MB,
        ram_free_mb=int(memory.available) // _MB,
        cpu_count=psutil.cpu_count(logical=True) or 1,
        on_ac=on_ac_power(),
    )
