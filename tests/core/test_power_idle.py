import pytest

from localdoc_finder.core import idle
from localdoc_finder.core.idle import IdleGate, SystemActivity
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.settings import IdleSettings, PowerSettings


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Env:
    """Mutable fake machine: power, CPU, input, fullscreen, GPU, chat."""

    def __init__(self) -> None:
        self.ac = True
        self.cpu = 1.0
        self.input_idle = 10_000.0
        self.fullscreen = False
        self.gpu: int | None = 0
        self.free_vram: int | None = 6000
        self.chat = False

    def on_ac(self) -> bool:
        return self.ac

    def cpu_percent(self) -> float:
        return self.cpu

    def seconds_since_input(self) -> float:
        return self.input_idle

    def fullscreen_app_active(self) -> bool:
        return self.fullscreen

    def gpu_utilization(self) -> int | None:
        return self.gpu

    def gpu_free_vram_mb(self) -> int | None:
        return self.free_vram


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def env() -> Env:
    return Env()


@pytest.fixture
def power(env: Env, clock: Clock) -> PowerGate:
    return PowerGate(PowerSettings(), probe=env.on_ac, clock=clock)


@pytest.fixture
def gate(power: PowerGate, env: Env, clock: Clock) -> IdleGate:
    return IdleGate(power, IdleSettings(), probe=env, clock=clock, chat_active=lambda: env.chat)


def settle(gate: IdleGate, clock: Clock, seconds: float = 200) -> None:
    gate.update()
    clock.now += seconds
    gate.update()


class TestPowerGate:
    def test_battery_blocks_indexing_but_allows_search(self, power: PowerGate, env: Env) -> None:
        env.ac = False
        power.update()
        assert power.ready_to_index() == (False, "battery, queue only")
        assert power.may_continue_indexing() == (False, "unplugged")
        assert power.search_allowed()
        assert power.search_on_cpu()
        assert not power.local_chat_allowed()

    def test_ac_must_settle(self, power: PowerGate, clock: Clock) -> None:
        power.update()
        assert power.on_ac
        assert power.ready_to_index() == (False, "waiting for AC to settle")
        clock.now += 121
        assert power.ready_to_index() == (True, "")

    def test_replugging_restarts_the_settle_timer(
        self, power: PowerGate, env: Env, clock: Clock
    ) -> None:
        power.update()
        clock.now += 200
        env.ac = False
        power.update()
        clock.now += 30
        assert power.unplugged_for() == 30
        env.ac = True
        power.update()
        assert power.unplugged_for() == 0
        assert power.ready_to_index()[0] is False

    def test_allow_battery_override(self, power: PowerGate, env: Env) -> None:
        env.ac = False
        power.update()
        assert power.ready_to_index(allow_battery=True) == (True, "")
        assert power.may_continue_indexing(allow_battery=True) == (True, "")

    def test_policy_can_be_disabled(self, env: Env, clock: Clock) -> None:
        env.ac = False
        gate = PowerGate(PowerSettings(require_ac_power=False), probe=env.on_ac, clock=clock)
        gate.update()
        assert gate.ready_to_index() == (True, "")
        assert gate.may_continue_indexing() == (True, "")

    def test_battery_options(self, env: Env, clock: Clock) -> None:
        env.ac = False
        settings = PowerSettings(
            search_on_battery=False, search_cpu_on_battery=False, chat_on_battery=True
        )
        gate = PowerGate(settings, probe=env.on_ac, clock=clock)
        assert not gate.search_allowed()
        assert not gate.search_on_cpu()
        assert gate.local_chat_allowed()
        env.ac = True
        assert gate.search_allowed()


class TestIdleGate:
    def test_ready_when_everything_is_quiet(self, gate: IdleGate, clock: Clock) -> None:
        settle(gate, clock)
        assert gate.ready() == (True, "")

    def test_battery_is_reported_first(self, gate: IdleGate, env: Env, clock: Clock) -> None:
        env.ac = False
        settle(gate, clock)
        assert gate.ready() == (False, "battery, queue only")
        assert not gate.on_ac

    def test_cpu_must_be_quiet_for_a_while(self, gate: IdleGate, env: Env, clock: Clock) -> None:
        env.cpu = 90
        settle(gate, clock)
        assert gate.ready() == (False, idle.REASON_CPU)
        env.cpu = 1
        gate.update()
        assert gate.ready() == (False, idle.REASON_CPU)  # quiet, but not for long enough
        clock.now += 61
        assert gate.ready() == (True, "")

    def test_user_fullscreen_chat_and_gpu_block(
        self, gate: IdleGate, env: Env, clock: Clock
    ) -> None:
        settle(gate, clock)
        env.input_idle = 5
        assert gate.ready() == (False, idle.REASON_USER)
        env.input_idle = 10_000
        env.fullscreen = True
        assert gate.ready() == (False, idle.REASON_FULLSCREEN)
        env.fullscreen = False
        env.chat = True
        assert gate.ready() == (False, idle.REASON_CHAT)
        env.chat = False
        env.gpu = 85
        assert gate.ready() == (False, "gpu busy (85%)")
        env.gpu = None  # no NVML: not a blocker
        assert gate.ready() == (True, "")

    def test_worker_may_continue_checks_power_chat_and_activity(
        self, gate: IdleGate, env: Env
    ) -> None:
        assert gate.worker_may_continue() == (True, "")
        env.input_idle = 1
        assert gate.worker_may_continue() == (False, idle.REASON_RETURNED)
        assert gate.worker_may_continue(respect_activity=False) == (True, "")
        env.input_idle = 10_000
        env.fullscreen = True
        assert gate.worker_may_continue() == (False, idle.REASON_FULLSCREEN)
        env.fullscreen = False
        env.chat = True
        assert gate.worker_may_continue(respect_activity=False) == (False, idle.REASON_CHAT)
        env.chat = False
        env.ac = False
        assert gate.worker_may_continue() == (False, "unplugged")
        assert gate.worker_may_continue(allow_battery=True) == (True, "")

    def test_worker_stops_when_something_else_takes_the_gpu_memory(
        self, gate: IdleGate, env: Env
    ) -> None:
        env.free_vram = 100
        ok, why = gate.worker_may_continue()
        assert not ok
        assert "gpu memory low" in why
        assert gate.worker_may_continue(respect_activity=False) == (True, "")  # CLI --now
        env.free_vram = None  # no NVML: not a blocker
        assert gate.worker_may_continue() == (True, "")


class TestTickArithmetic:
    def test_elapsed_ticks_is_plain_subtraction_without_wrap(self) -> None:
        assert idle.elapsed_ticks(10_000, 4_000) == 6_000

    def test_elapsed_ticks_survives_the_32_bit_wrap(self) -> None:
        # ~24.8 days of uptime: the signed 32-bit counter wrapped; 49.7 days: the unsigned one
        assert idle.elapsed_ticks(5, 0xFFFFFFFF - 994) == 1000
        assert idle.elapsed_ticks(0x80000000 + 500, 0x7FFFFFFF - 499) == 1000


class TestSystemActivity:
    """Smoke tests against the real machine: values only need to be sane."""

    def test_probes_return_sane_values(self) -> None:
        probe = SystemActivity()
        assert 0 <= probe.cpu_percent() <= 100
        assert probe.seconds_since_input() >= 0
        assert isinstance(probe.fullscreen_app_active(), bool)
        util = probe.gpu_utilization()
        assert util is None or 0 <= util <= 100
        free = probe.gpu_free_vram_mb()
        assert free is None or free >= 0

    def test_gpu_probe_without_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "pynvml", None)
        probe = SystemActivity()
        assert probe.gpu_utilization() is None
        assert probe.gpu_free_vram_mb() is None
        assert idle.gpu_present() in {True, False}

    def test_default_probe_is_the_system(self, power: PowerGate) -> None:
        assert isinstance(IdleGate(power, IdleSettings())._probe, SystemActivity)


class TestNvmlHandleIsReused:
    def test_nvml_is_initialised_once_across_many_ticks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        from types import SimpleNamespace

        inits: list[int] = []
        fake = SimpleNamespace(
            nvmlInit=lambda: inits.append(1),
            nvmlShutdown=lambda: inits.append(-1),
            nvmlDeviceGetHandleByIndex=lambda _i: "handle",
            nvmlDeviceGetUtilizationRates=lambda _h: SimpleNamespace(gpu=7),
            nvmlDeviceGetMemoryInfo=lambda _h: SimpleNamespace(free=2048 * 1024 * 1024),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake)
        probe = SystemActivity()
        for _ in range(5):
            assert probe.gpu_utilization() == 7
            assert probe.gpu_free_vram_mb() == 2048
        assert inits == [1]  # no init/shutdown per tick

    def test_a_failed_nvml_init_is_not_retried_every_tick(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        from types import SimpleNamespace

        attempts: list[int] = []

        def broken() -> None:
            attempts.append(1)
            raise OSError("no driver")

        monkeypatch.setitem(sys.modules, "pynvml", SimpleNamespace(nvmlInit=broken))
        probe = SystemActivity()
        for _ in range(4):
            assert probe.gpu_utilization() is None
        assert attempts == [1]
