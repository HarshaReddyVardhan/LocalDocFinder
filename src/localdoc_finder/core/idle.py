"""Idle gate (on top of power): CPU quiet, no user input, GPU free, nothing fullscreen, no chat."""

import ctypes
import logging
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any, Protocol

import psutil

from localdoc_finder.core.models.hardware import probe_gpu
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.settings import IdleSettings

logger = logging.getLogger(__name__)

REASON_CPU = "cpu busy"
REASON_USER = "user active"
REASON_FULLSCREEN = "fullscreen app"
REASON_CHAT = "chat active"
REASON_RETURNED = "user returned"

_NEVER = 1e9
_MB = 1024 * 1024
_TICK_MASK = 0xFFFFFFFF  # LASTINPUTINFO.dwTime is a 32-bit tick count that wraps every ~49 days
# SHQueryUserNotificationState values that mean "do not disturb": a full-screen or Direct3D
# program, or a presentation (this is what Focus Assist reports for those cases).
_BUSY_NOTIFICATION_STATES = frozenset({2, 3, 4})
_SHELL_CLASSES = ("Progman", "WorkerW", "Shell_TrayWnd")
_MONITOR_DEFAULTTONEAREST = 2


class ActivityProbe(Protocol):
    """Machine activity signals; replaced by fakes in tests."""

    def cpu_percent(self) -> float: ...

    def seconds_since_input(self) -> float: ...

    def fullscreen_app_active(self) -> bool: ...

    def gpu_utilization(self) -> int | None: ...

    def gpu_free_vram_mb(self) -> int | None: ...


def elapsed_ticks(now: int, last: int) -> int:
    """Milliseconds between two 32-bit tick counts, correct across the wrap-around."""
    return (now - last) & _TICK_MASK


class _LastInputInfo(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class SystemActivity:
    """Real Windows probes (psutil, user32, NVML)."""

    def __init__(self) -> None:
        psutil.cpu_percent(None)  # prime the counter
        self._nvml: tuple[Any, Any] | None = None
        self._nvml_failed = False

    def cpu_percent(self) -> float:
        return float(psutil.cpu_percent(None))

    def seconds_since_input(self) -> float:
        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined,unused-ignore]
            return _NEVER
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
        kernel32.GetTickCount64.restype = ctypes.c_uint64
        now = int(kernel32.GetTickCount64()) & _TICK_MASK
        return elapsed_ticks(now, int(info.dwTime)) / 1000.0

    def fullscreen_app_active(self) -> bool:
        """True for a full-screen window, a Direct3D game or a presentation."""
        return self._notification_state_busy() or self._foreground_covers_monitor()

    @staticmethod
    def _notification_state_busy() -> bool:
        try:
            state = ctypes.c_int(0)
            result = ctypes.windll.shell32.SHQueryUserNotificationState(  # type: ignore[attr-defined,unused-ignore]
                ctypes.byref(state)
            )
        except Exception:  # an old Windows without the API: fall back on the window check
            logger.debug("idle: notification state unavailable", exc_info=True)
            return False
        return result == 0 and state.value in _BUSY_NOTIFICATION_STATES

    @staticmethod
    def _foreground_covers_monitor() -> bool:
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined,unused-ignore]
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False
            class_name = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, class_name, 64)
            if class_name.value in _SHELL_CLASSES:
                return False
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            monitor = _MonitorInfo()
            monitor.cbSize = ctypes.sizeof(_MonitorInfo)
            handle = user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
            user32.GetMonitorInfoW(handle, ctypes.byref(monitor))
            m = monitor.rcMonitor
            return bool(
                rect.left <= m.left
                and rect.top <= m.top
                and rect.right >= m.right
                and rect.bottom >= m.bottom
            )
        except Exception:  # a failing probe must never block indexing forever
            logger.debug("idle: fullscreen probe failed", exc_info=True)
            return False

    def _nvml_handle(self) -> tuple[Any, Any] | None:
        """The first GPU's NVML handle, initialised once (init/shutdown per tick is wasteful)."""
        if self._nvml_failed:
            return None
        if self._nvml is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            except Exception:  # no NVIDIA GPU or driver
                logger.debug("idle: gpu probe unavailable", exc_info=True)
                self._nvml_failed = True
                return None
        return self._nvml

    def gpu_utilization(self) -> int | None:
        nvml = self._nvml_handle()
        if nvml is None:
            return None
        module, handle = nvml
        try:
            return int(module.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:  # the driver went away mid-run
            logger.debug("idle: gpu utilisation read failed", exc_info=True)
            return None

    def gpu_free_vram_mb(self) -> int | None:
        nvml = self._nvml_handle()
        if nvml is None:
            return None
        module, handle = nvml
        try:
            return int(module.nvmlDeviceGetMemoryInfo(handle).free) // _MB
        except Exception:
            logger.debug("idle: gpu memory read failed", exc_info=True)
            return None


class IdleGate:
    """Decides whether a worker may start (full gate) or continue (cheap re-check)."""

    def __init__(
        self,
        power: PowerGate,
        settings: IdleSettings,
        probe: ActivityProbe | None = None,
        clock: Callable[[], float] = time.monotonic,
        chat_active: Callable[[], bool] = lambda: False,
    ) -> None:
        self._power = power
        self._settings = settings
        self._probe = probe or SystemActivity()
        self._clock = clock
        self._chat_active = chat_active
        self._cpu_quiet_since: float | None = None

    def update(self) -> None:
        """Sample AC state and CPU load; call every few seconds."""
        self._power.update()
        if self._probe.cpu_percent() < self._settings.cpu_percent:
            if self._cpu_quiet_since is None:
                self._cpu_quiet_since = self._clock()
        else:
            self._cpu_quiet_since = None

    @property
    def on_ac(self) -> bool:
        return self._power.on_ac

    def ready(self, allow_battery: bool = False) -> tuple[bool, str]:
        """May a worker start now? Returns ``(ok, reason when not)``."""
        reason = self._start_blocker(allow_battery)
        return not reason, reason

    def _cpu_not_quiet(self) -> bool:
        quiet = self._cpu_quiet_since
        return quiet is None or self._clock() - quiet < self._settings.cpu_seconds

    def _user_recently_active(self) -> bool:
        return self._probe.seconds_since_input() < self._settings.no_input_seconds

    def _start_blocker(self, allow_battery: bool) -> str:
        ok, reason = self._power.ready_to_index(allow_battery)
        if not ok:
            return reason
        checks = (
            (self._cpu_not_quiet, REASON_CPU),
            (self._user_recently_active, REASON_USER),
            (self._probe.fullscreen_app_active, REASON_FULLSCREEN),
            (self._chat_active, REASON_CHAT),
        )
        for blocked, why in checks:
            if blocked():
                return why
        util = self._probe.gpu_utilization()
        if util is not None and util > self._settings.gpu_max_util_percent:
            return f"gpu busy ({util}%)"
        return ""

    def worker_may_continue(
        self, allow_battery: bool = False, respect_activity: bool = True
    ) -> tuple[bool, str]:
        """Cheap check before every batch: power, chat, then user activity.

        The worker itself burns CPU, so the CPU-quiet test is not repeated here.
        """
        reason = self._continue_blocker(allow_battery, respect_activity)
        return not reason, reason

    def _continue_blocker(self, allow_battery: bool, respect_activity: bool) -> str:
        ok, reason = self._power.may_continue_indexing(allow_battery)
        if not ok:
            return reason
        if self._chat_active():
            return REASON_CHAT
        if not respect_activity:
            return ""
        if self._probe.seconds_since_input() < self._settings.worker_yield_input_seconds:
            return REASON_RETURNED
        if self._probe.fullscreen_app_active():
            return REASON_FULLSCREEN
        return self._gpu_memory_blocker()

    def _gpu_memory_blocker(self) -> str:
        free = self._probe.gpu_free_vram_mb()
        if free is not None and free < self._settings.min_free_vram_mb:
            return f"gpu memory low ({free} MB free)"  # something else needs the card
        return ""


def gpu_present() -> bool:
    return probe_gpu() is not None
