"""Hardware probe: GPU/VRAM, RAM, CPU, power source and current load. Every probe degrades."""

import logging
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
CPU_SAMPLE_SECONDS = 0.5  # psutil needs a window; a zero window reports 0% on first call


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


@dataclass(frozen=True)
class Load:
    """How busy the machine is right now, in percent."""

    cpu_percent: float
    gpu_percent: int | None  # None without a readable NVIDIA GPU


def probe_gpu_busy() -> int | None:
    """Percent of the last sample period the first NVIDIA GPU was busy, or ``None``."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # NVML missing, no driver or no device: all mean "no usable GPU"
        logger.debug("hardware: GPU load unavailable", exc_info=True)
        return None


def probe_load() -> Load:
    """CPU and GPU load; blocks for ``CPU_SAMPLE_SECONDS`` to measure the CPU."""
    return Load(
        cpu_percent=float(psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS)),
        gpu_percent=probe_gpu_busy(),
    )


def on_ac_power() -> bool:
    """True if plugged in. No battery at all (a desktop) counts as plugged in.

    Anything unclear (a probe error, an unknown state) counts as unplugged: indexing never runs
    on battery, so a guess must go the safe way.
    """
    try:
        battery = psutil.sensors_battery()
    except Exception:  # psutil can raise on odd platforms
        logger.debug("hardware: battery probe failed; assuming unplugged", exc_info=True)
        return False
    if battery is None:
        return True
    return battery.power_plugged is True


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
