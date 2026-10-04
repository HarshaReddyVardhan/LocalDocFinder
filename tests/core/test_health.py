from tests.core.conftest import Env

from localdoc_finder.core import health
from localdoc_finder.core.health import collect_health, format_health, month_start
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.hardware import Hardware, Load
from localdoc_finder.core.models.registry import ModelRegistry
from localdoc_finder.core.providers.base import CAP_COMPLETION, ModelInfo, ProviderError
from localdoc_finder.core.store.sqlite import CHAT_LOCK

GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
LOAD = Load(cpu_percent=23.4, gpu_percent=41)
NOW = 1_780_000_000.0  # mid-month


class Provider:
    name = "ollama"

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("qwen3.5:9b", "ollama", capabilities=frozenset({CAP_COMPLETION}))]


def registry() -> ModelRegistry:
    reg = ModelRegistry(load_catalog(), [Provider()], hardware_probe=lambda: GPU)
    reg.refresh()
    return reg


def test_snapshot_collects_every_probe(env: Env) -> None:
    env.state.manifest_set("a", 1, 1, "h")
    env.state.enqueue("b", delay=0)
    env.state.enqueue("c", delay=9999)
    env.state.set_meta("last_reconcile", str(NOW - 3600))
    env.state.acquire_lock(CHAT_LOCK, "me", 600)
    env.state.record_usage("openrouter", "m", 100, 20, 0.25)
    snap = collect_health(
        env.state, env.store, registry(), lambda: ["qwen3.5:9b"],
        hardware=GPU, load=LOAD, budget_usd=1.0, clock=lambda: NOW,
    )  # fmt: skip
    assert (snap.indexed_files, snap.queue_total, snap.queue_due) == (1, 2, 1)
    assert snap.loaded_models == ["qwen3.5:9b"]
    assert snap.chat_active
    assert snap.last_reconcile == NOW - 3600
    assert snap.month_spend_usd == 0.25
    assert snap.models is not None
    assert snap.models.resolutions["chat"].model == "qwen3.5:9b"
    assert not snap.over_budget


def test_unreachable_server_means_no_loaded_models(env: Env) -> None:
    def down() -> list[str]:
        raise ProviderError("down")

    snap = collect_health(env.state, env.store, registry(), down, hardware=GPU, load=LOAD)
    assert snap.loaded_models == []
    assert snap.last_reconcile is None
    assert snap.budget_usd is None


def test_formatting_covers_gpu_budget_and_usage(env: Env) -> None:
    env.state.record_usage("openrouter", "gpt-x", 1000, 200, 1.5)
    snap = collect_health(
        env.state, env.store, registry(), lambda: [],
        hardware=GPU, load=LOAD, budget_usd=1.0,
    )  # fmt: skip
    text = format_health(snap)
    assert "CPU use         : 23% (8 threads)" in text
    assert "GPU use         : 41%" in text
    assert "1192/8192 MB used (RTX 2070)" in text
    assert "none (VRAM free)" in text
    assert "last reconcile  : never" in text
    assert "cloud spend     : $1.5000 of $1.00 this month" in text
    assert "budget reached" in text
    assert "openrouter/gpt-x: 1000 in, 200 out, $1.5000" in text


def test_formatting_without_a_gpu_or_budget(env: Env) -> None:
    cpu = Hardware(None, 0, 0, 16000, 8000, 4, False)
    no_gpu = Load(cpu_percent=5.0, gpu_percent=None)
    snap = collect_health(
        env.state, env.store, registry(), lambda: ["m"], hardware=cpu, load=no_gpu
    )
    snap = health.HealthSnapshot(**{**snap.__dict__, "last_reconcile": NOW})
    text = format_health(snap)
    assert "no NVIDIA GPU" in text
    assert "GPU use" not in text
    assert "battery (indexing paused)" in text
    assert "budget reached" not in text
    assert "last reconcile  : 20" in text


def test_month_start_is_the_first_of_the_month_midnight() -> None:
    from datetime import datetime

    start = datetime.fromtimestamp(month_start(NOW))
    assert (start.day, start.hour, start.minute, start.second) == (1, 0, 0, 0)
    assert month_start(NOW) <= NOW
