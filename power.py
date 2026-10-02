"""Power policy and idle gate (Windows).

Power: indexing only runs on AC. Idle: CPU quiet, no recent input, GPU free, nothing fullscreen.
The worker itself burns CPU, so it re-checks only power / input / fullscreen between batches
(`worker_may_continue`); the full gate (`IdleGate`) is for the watcher deciding whether to START it.
"""
import ctypes
import time
from ctypes import wintypes
from typing import Optional, Tuple

import psutil

import indexer_config as cfg


def on_ac_power() -> bool:
    """True if plugged in. No battery (desktop) counts as plugged in."""
    try:
        b = psutil.sensors_battery()
    except Exception:
        return True
    if b is None or b.power_plugged is None:
        return True
    return bool(b.power_plugged)


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def seconds_since_input() -> float:
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 1e9
    millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0, millis) / 1000.0


def fullscreen_app_active() -> bool:
    """True if the foreground window covers its whole monitor (game, video, presentation)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        # Ignore the desktop / shell windows.
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, cls, 64)
        if cls.value in ("Progman", "WorkerW", "Shell_TrayWnd"):
            return False
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        MONITOR_DEFAULTTONEAREST = 2

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        mon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        user32.GetMonitorInfoW(mon, ctypes.byref(mi))
        m = mi.rcMonitor
        return (rect.left <= m.left and rect.top <= m.top
                and rect.right >= m.right and rect.bottom >= m.bottom)
    except Exception:
        return False


def gpu_utilization() -> Optional[int]:
    """GPU util % (None if NVML is unavailable)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def worker_may_continue(allow_battery: bool = False, respect_activity: bool = True) -> Tuple[bool, str]:
    """Cheap check the worker runs before every batch (power, then user activity)."""
    if cfg.REQUIRE_AC_POWER and not allow_battery and not on_ac_power():
        return False, "unplugged"
    if respect_activity:
        if seconds_since_input() < cfg.WORKER_YIELD_INPUT_SECONDS:
            return False, "user returned"
        if fullscreen_app_active():
            return False, "fullscreen app"
    return True, ""


class IdleGate:
    """Tracks AC stability and CPU idleness; call update() periodically, then ready()."""

    def __init__(self):
        self._ac_since: Optional[float] = None
        self._cpu_quiet_since: Optional[float] = None
        psutil.cpu_percent(None)  # prime

    def update(self) -> None:
        now = time.monotonic()
        if on_ac_power():
            if self._ac_since is None:
                self._ac_since = now
        else:
            self._ac_since = None
        if psutil.cpu_percent(None) < cfg.IDLE_CPU_PERCENT:
            if self._cpu_quiet_since is None:
                self._cpu_quiet_since = now
        else:
            self._cpu_quiet_since = None

    @property
    def on_ac(self) -> bool:
        return self._ac_since is not None

    def ready(self, allow_battery: bool = False) -> Tuple[bool, str]:
        now = time.monotonic()
        if cfg.REQUIRE_AC_POWER and not allow_battery:
            if self._ac_since is None:
                return False, "battery, queue only"
            if now - self._ac_since < cfg.AC_SETTLE_SECONDS:
                return False, "waiting for AC to settle"
        if self._cpu_quiet_since is None or now - self._cpu_quiet_since < cfg.IDLE_CPU_SECONDS:
            return False, "cpu busy"
        if seconds_since_input() < cfg.IDLE_NO_INPUT_SECONDS:
            return False, "user active"
        if fullscreen_app_active():
            return False, "fullscreen app"
        util = gpu_utilization()
        if util is not None and util > cfg.IDLE_GPU_MAX_UTIL_PERCENT:
            return False, f"gpu busy ({util}%)"
        return True, ""
