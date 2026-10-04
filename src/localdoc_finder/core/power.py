"""Power policy: indexing only on AC; search and chat degrade gracefully on battery."""

import time
from collections.abc import Callable

from localdoc_finder.core.models.hardware import on_ac_power
from localdoc_finder.core.settings import PowerSettings

REASON_BATTERY = "battery, queue only"
REASON_UNPLUGGED = "unplugged"
REASON_SETTLING = "waiting for AC to settle"


class PowerGate:
    """Tracks how long the machine has been on AC. Call ``update`` periodically."""

    def __init__(
        self,
        settings: PowerSettings,
        probe: Callable[[], bool] = on_ac_power,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._probe = probe
        self._clock = clock
        self._ac_since: float | None = None
        self._unplugged_since: float | None = None

    def update(self) -> None:
        now = self._clock()
        if self._probe():
            if self._ac_since is None:
                self._ac_since = now
            self._unplugged_since = None
        else:
            self._ac_since = None
            if self._unplugged_since is None:
                self._unplugged_since = now

    @property
    def on_ac(self) -> bool:
        return self._ac_since is not None

    def unplugged_for(self) -> float:
        """Seconds since the charger was removed (0 while plugged in)."""
        if self._unplugged_since is None:
            return 0.0
        return self._clock() - self._unplugged_since

    def ready_to_index(self, allow_battery: bool = False) -> tuple[bool, str]:
        """May a worker *start*? Needs AC that has been stable for ``ac_settle_seconds``."""
        if not self._settings.require_ac_power or allow_battery:
            return True, ""
        if self._ac_since is None:
            return False, REASON_BATTERY
        if self._clock() - self._ac_since < self._settings.ac_settle_seconds:
            return False, REASON_SETTLING
        return True, ""

    # Policies that depend on the *current* power state, not on stability.
    def may_continue_indexing(self, allow_battery: bool = False) -> tuple[bool, str]:
        """Checked by the worker before every batch; unplugging stops it within one batch."""
        if self._settings.require_ac_power and not allow_battery and not self._probe():
            return False, REASON_UNPLUGGED
        return True, ""

    def search_allowed(self) -> bool:
        return self._settings.search_on_battery or self._probe()

    def search_on_cpu(self) -> bool:
        """Embed queries with ``num_gpu=0`` when unplugged so the GPU stays idle."""
        return self._settings.search_cpu_on_battery and not self._probe()

    def local_chat_allowed(self) -> bool:
        return self._settings.chat_on_battery or self._probe()
