"""Health dashboard data: CPU/GPU load, queue, index size, VRAM and loaded models, usage, cost."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from localdoc_finder.core.models.hardware import Hardware, Load, probe_hardware, probe_load
from localdoc_finder.core.models.registry import ModelRegistry, Report
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.store.lance import DOCUMENTS, LanceStore
from localdoc_finder.core.store.sqlite import CHAT_LOCK, StateDb, UsageTotal

logger = logging.getLogger(__name__)

LoadedModels = Callable[[], list[str]]


@dataclass(frozen=True)
class HealthSnapshot:
    hardware: Hardware
    load: Load
    indexed_files: int
    queue_total: int
    queue_due: int
    chunks: int
    documents: int
    last_reconcile: float | None
    loaded_models: list[str]
    chat_active: bool
    usage: list[UsageTotal] = field(default_factory=list)
    month_spend_usd: float = 0.0
    budget_usd: float | None = None
    models: Report | None = None

    @property
    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.month_spend_usd >= self.budget_usd


def month_start(now: float) -> float:
    moment = datetime.fromtimestamp(now)  # local calendar month
    return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def collect_health(
    state: StateDb,
    store: LanceStore,
    registry: ModelRegistry,
    loaded_models: LoadedModels,
    *,
    hardware: Hardware | None = None,
    load: Load | None = None,
    budget_usd: float | None = None,
    clock: Callable[[], float] = time.time,
) -> HealthSnapshot:
    """Gather everything the dashboard shows. Each probe degrades instead of failing."""
    hw = hardware or probe_hardware()
    try:
        loaded = loaded_models()
    except ProviderError:
        logger.debug("health: cannot list loaded models", exc_info=True)
        loaded = []
    now = clock()
    last = state.get_meta("last_reconcile")
    return HealthSnapshot(
        hardware=hw,
        load=load or probe_load(),
        indexed_files=state.manifest_count(),
        queue_total=state.queue_size(),
        queue_due=state.queue_size(due_only=True),
        chunks=store.count(),
        documents=store.count(DOCUMENTS),
        last_reconcile=float(last) if last and float(last) > 0 else None,
        loaded_models=loaded,
        chat_active=state.lock_held(CHAT_LOCK),
        usage=state.usage_totals(month_start(now)),
        month_spend_usd=state.spend_since(month_start(now)),
        budget_usd=budget_usd,
        models=registry.report(hw),
    )


def format_health(snapshot: HealthSnapshot) -> str:
    hw = snapshot.hardware
    vram = (
        f"{hw.vram_total_mb - hw.vram_free_mb}/{hw.vram_total_mb} MB used ({hw.gpu_name})"
        if hw.has_gpu
        else "no NVIDIA GPU"
    )
    reconcile = (
        datetime.fromtimestamp(snapshot.last_reconcile).strftime("%Y-%m-%d %H:%M")
        if snapshot.last_reconcile
        else "never"
    )
    loaded = ", ".join(snapshot.loaded_models) or "none (VRAM free)"
    lines = [
        f"power           : {'AC' if hw.on_ac else 'battery (indexing paused)'}",
        f"CPU use         : {snapshot.load.cpu_percent:.0f}% ({hw.cpu_count} threads)",
    ]
    if snapshot.load.gpu_percent is not None:
        lines.append(f"GPU use         : {snapshot.load.gpu_percent}%")
    lines += [
        f"VRAM            : {vram}",
        f"loaded models   : {loaded}",
        f"chat session    : {'active' if snapshot.chat_active else 'idle'}",
        f"indexed files   : {snapshot.indexed_files} ({snapshot.chunks} chunks, "
        f"{snapshot.documents} documents)",
        f"queue           : {snapshot.queue_total} total, {snapshot.queue_due} due",
        f"last reconcile  : {reconcile}",
    ]
    budget = f" of ${snapshot.budget_usd:.2f}" if snapshot.budget_usd is not None else ""
    lines.append(f"cloud spend     : ${snapshot.month_spend_usd:.4f}{budget} this month")
    if snapshot.over_budget:
        lines.append("                  budget reached: cloud calls are paused")
    for usage in snapshot.usage:
        lines.append(
            f"  {usage.provider}/{usage.model}: {usage.prompt_tokens} in, "
            f"{usage.completion_tokens} out, ${usage.cost_usd:.4f}"
        )
    return "\n".join(lines)
