"""Idle gate (on top of power): CPU quiet, no user input, GPU free, nothing fullscreen, no chat."""

import ctypes
import logging
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Protocol

import psutil

from vector_embed.core.models.hardware import probe_gpu
from vector_embed.core.power import PowerGate
from vector_embed.core.settings import IdleSettings

logger = logging.getLogger(__name__)

REASON_CPU = "cpu busy"
REASON_USER = "user active"
REASON_FULLSCREEN = "fullscreen app"
REASON_CHAT = "chat active"
REASON_RETURNED = "user returned"

_NEVER = 1e9
_SHELL_CLASSES = ("Progman", "WorkerW", "Shell_TrayWnd")
_MONITOR_DEFAULTTONEAREST = 2


class ActivityProbe(Protocol):
    """Machine activity signals; replaced by fakes in tests."""

    def cpu_percent(self) -> float: ...

    def seconds_since_input(self) -> float: ...

    def fullscreen_app_active(self) -> bool: ...

    def gpu_utilization(self) -> int | None: ...


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

    def cpu_percent(self) -> float:
        return float(psutil.cpu_percent(None))

    def seconds_since_input(self) -> float:
        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined,unused-ignore]
            return _NEVER
        millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime  # type: ignore[attr-defined,unused-ignore]
        return float(max(0, millis)) / 1000.0

    def fullscreen_app_active(self) -> bool:
        """True if the foreground window covers its whole monitor (game, video, presentation)."""
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

    def gpu_utilization(self) -> int | None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # no NVIDIA GPU or driver
            logger.debug("idle: gpu probe unavailable", exc_info=True)
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
        return ""


def gpu_present() -> bool:
    return probe_gpu() is not None
