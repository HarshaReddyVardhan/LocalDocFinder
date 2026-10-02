"""Starter model set: what to download on a machine that has nothing installed yet.

Pure decision logic. Unlike the runtime resolver (which fits against *free* VRAM) this checks a
model against the machine's *total* capacity, since nothing of ours is loaded at setup time. The
embedder and chat model are never resident together, so each role is fitted on its own.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_EMBED, Catalog
from vector_embed.core.models.fit import budget_mb, fits
from vector_embed.core.models.hardware import Hardware

STARTER_ROLES = (ROLE_EMBED, ROLE_CHAT)


@dataclass(frozen=True)
class StarterPick:
    role: str
    model: str
    reason: str
    download_mb: int
    downgrade: str | None  # next smaller model that fits, offered if the speed test says "slow"


@dataclass(frozen=True)
class StarterPlan:
    picks: tuple[StarterPick, ...]
    missing_roles: tuple[str, ...]  # roles for which no catalog model fits this machine

    def pick_for(self, role: str) -> StarterPick | None:
        return next((p for p in self.picks if p.role == role), None)

    @property
    def models(self) -> tuple[str, ...]:
        """Distinct model names to download, in role order."""
        return tuple(dict.fromkeys(p.model for p in self.picks))

    @property
    def total_download_mb(self) -> int:
        sizes = {p.model: p.download_mb for p in self.picks}
        return sum(sizes.values())


def _fitting(catalog: Catalog, role: str, hardware: Hardware) -> list[str]:
    """Models for ``role`` that fit, best-first; ones without a catalog entry are skipped."""
    budget = budget_mb(hardware, total=True)
    return [
        name
        for name in catalog.preferences(role)
        if (entry := catalog.entry(name)) is not None
        and fits(entry, entry.vram_mb, hardware, budget)
    ]


def _reason(catalog: Catalog, name: str, hardware: Hardware) -> str:
    need = catalog.vram_mb(name)
    if hardware.has_gpu:
        return f"needs ~{need} MB, fits your {hardware.vram_total_mb} MB of VRAM"
    return f"CPU-only: needs ~{need} MB, fits within half of your {hardware.ram_total_mb} MB RAM"


def _downgrade(catalog: Catalog, name: str, fitting: Sequence[str]) -> str | None:
    """The best fitting model that is smaller than ``name``."""
    size = catalog.vram_mb(name) or 0
    return next((n for n in fitting if (catalog.vram_mb(n) or 0) < size), None)


def pick_starter(
    catalog: Catalog, hardware: Hardware, roles: Sequence[str] = STARTER_ROLES
) -> StarterPlan:
    """Choose the first preferred model per role that fits ``hardware``."""
    picks: list[StarterPick] = []
    missing: list[str] = []
    for role in roles:
        fitting = _fitting(catalog, role, hardware)
        if not fitting:
            missing.append(role)
            continue
        name = fitting[0]
        entry = catalog.entry(name)
        picks.append(
            StarterPick(
                role=role,
                model=name,
                reason=_reason(catalog, name, hardware),
                download_mb=entry.download_mb if entry else 0,
                downgrade=_downgrade(catalog, name, fitting),
            )
        )
    return StarterPlan(tuple(picks), tuple(missing))
