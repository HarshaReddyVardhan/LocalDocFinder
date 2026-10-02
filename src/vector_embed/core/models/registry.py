"""Model registry: discovery, role resolution, and recommendations.

Features ask for a *role* (``chat``, ``embed``, ``caption`` ...), never a model name. The
resolver picks the best installed model for the role that fits free VRAM; the user can
override any role. The ``embed`` role is pinned because vectors from different models cannot
be mixed: changing it needs an explicit re-index.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from vector_embed.core.models.catalog import (
    ROLE_CAPTION,
    ROLE_EMBED,
    ROLES,
    Catalog,
)
from vector_embed.core.models.hardware import Hardware, probe_hardware
from vector_embed.core.providers.base import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    CAP_VISION,
    ModelInfo,
    ProviderError,
)
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_SIZE_TO_VRAM = 1.15  # resident size is a little over the file size once the KV cache exists
_CPU_RAM_FRACTION = 0.5  # share of free RAM a CPU-only machine may devote to a model
_REFRESHED_KEY = "models_refreshed_at"
_SECONDS_PER_FILE = 0.5  # rough embedding throughput used for the re-index estimate

FLAG_OK = "ok"
FLAG_WARNING = "warning"
FLAG_UNUSED = "unused"


class Discoverable(Protocol):
    name: str

    def list_models(self) -> list[ModelInfo]: ...


@dataclass(frozen=True)
class Resolution:
    role: str
    model: str | None
    reason: str


@dataclass(frozen=True)
class Flag:
    kind: str
    message: str


@dataclass(frozen=True)
class ModelRow:
    info: ModelInfo
    roles: tuple[str, ...]
    flags: tuple[Flag, ...]


@dataclass(frozen=True)
class Recommendation:
    role: str
    model: str
    reason: str

    @property
    def pull_command(self) -> str:
        return f"ollama pull {self.model}"


@dataclass(frozen=True)
class ReindexNotice:
    files: int
    estimated_seconds: float
    message: str


@dataclass(frozen=True)
class Report:
    hardware: Hardware
    resolutions: dict[str, Resolution]
    rows: list[ModelRow] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)


def _canonical(name: str) -> str:
    return name.removesuffix(":latest")


class ModelRegistry:
    def __init__(
        self,
        catalog: Catalog,
        providers: Sequence[Discoverable],
        state: StateDb | None = None,
        *,
        hardware_probe: Callable[[], Hardware] = probe_hardware,
        overrides: Mapping[str, str] | None = None,
        pinned_embed: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._catalog = catalog
        self._providers = list(providers)
        self._state = state
        self._probe = hardware_probe
        self._overrides = dict(overrides or {})
        self._pinned_embed = pinned_embed
        self._clock = clock
        self._installed: list[ModelInfo] = []

    # ------------------------------------------------------------------ discovery
    def refresh(self) -> list[ModelInfo]:
        """Re-discover models from every provider. An unreachable provider is skipped."""
        found: list[ModelInfo] = []
        for provider in self._providers:
            try:
                models = provider.list_models()
            except ProviderError:
                logger.warning("models: cannot list models from %s", provider.name)
                continue
            found.extend(models)
            if self._state is not None:
                self._state.replace_models(
                    provider.name,
                    [
                        {
                            "name": m.name,
                            "size_bytes": m.size_bytes,
                            "parameter_size": m.parameter_size,
                            "quantization": m.quantization,
                            "context_length": m.context_length,
                            "capabilities": sorted(m.capabilities),
                        }
                        for m in models
                    ],
                )
        self._installed = found
        if self._state is not None:
            self._state.set_meta(_REFRESHED_KEY, str(self._clock()))
        return found

    def refresh_if_stale(self, max_age_seconds: float = 3600) -> bool:
        """Refresh when the last discovery is older than ``max_age_seconds``."""
        last = self._state.get_meta(_REFRESHED_KEY) if self._state else None
        if last is not None and self._clock() - float(last) < max_age_seconds and self._installed:
            return False
        self.refresh()
        return True

    @property
    def installed(self) -> list[ModelInfo]:
        return list(self._installed)

    def _lookup(self) -> dict[str, ModelInfo]:
        return {_canonical(m.name): m for m in self._installed}

    # ------------------------------------------------------------------ resolving
    def _budget_mb(self, hardware: Hardware) -> int:
        if hardware.has_gpu:
            return hardware.vram_free_mb
        return int(hardware.ram_free_mb * _CPU_RAM_FRACTION)

    def _needs_mb(self, info: ModelInfo) -> int | None:
        known = self._catalog.vram_mb(info.name) or self._catalog.vram_mb(_canonical(info.name))
        if known is not None:
            return known
        if info.size_bytes:
            return int(info.size_bytes / _MB * _SIZE_TO_VRAM)
        return None

    def _fits(self, name: str, info: ModelInfo | None, hardware: Hardware) -> bool:
        entry = self._catalog.models.get(name)
        if not hardware.has_gpu and entry is not None and not entry.cpu_ok:
            return False
        needed = self._catalog.vram_mb(name)
        if needed is None and info is not None:
            needed = self._needs_mb(info)
        return needed is None or needed <= self._budget_mb(hardware)

    @staticmethod
    def _suits_role(role: str, info: ModelInfo) -> bool:
        caps = info.capabilities
        if not caps:
            return True  # capabilities unknown; do not exclude
        if role == ROLE_EMBED:
            return CAP_EMBEDDING in caps
        if role == ROLE_CAPTION:
            return CAP_VISION in caps
        return CAP_COMPLETION in caps and CAP_EMBEDDING not in caps

    def resolve(self, role: str, hardware: Hardware | None = None) -> Resolution:
        """Best installed model for ``role`` that fits; ``model`` is ``None`` if nothing does."""
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        hw = hardware or self._probe()
        installed = self._lookup()

        pinned = self._pinned_embed if role == ROLE_EMBED else None
        chosen = pinned or self._overrides.get(role)
        if chosen:
            info = installed.get(_canonical(chosen))
            if info is not None:
                return Resolution(role, info.name, "pinned" if pinned else "override")
            return Resolution(role, None, f"{chosen} is configured but not installed")

        for name in self._catalog.preferences(role):
            info = installed.get(_canonical(name))
            if info is not None and self._fits(name, info, hw):
                return Resolution(role, info.name, "preferred")

        fallbacks = [
            m
            for m in self._installed
            if self._suits_role(role, m)
            and _canonical(m.name) not in self._catalog.warnings
            and m.name not in self._catalog.preferences(role)
            and self._fits(m.name, m, hw)
        ]
        if fallbacks:
            best = max(fallbacks, key=lambda m: m.size_bytes or 0)
            return Resolution(role, best.name, "fallback: best installed model with the capability")
        return Resolution(role, None, "no installed model fits")

    def resolve_all(self, hardware: Hardware | None = None) -> dict[str, Resolution]:
        hw = hardware or self._probe()
        return {role: self.resolve(role, hw) for role in ROLES}

    # ------------------------------------------------------------------ recommendations
    def recommendations(
        self, resolutions: Mapping[str, Resolution], hardware: Hardware
    ) -> list[Recommendation]:
        installed = self._lookup()
        out: list[Recommendation] = []
        for role, resolution in resolutions.items():
            prefs = self._catalog.preferences(role)
            current = (
                prefs.index(resolution.model)
                if resolution.model in prefs
                else len(prefs)
                if resolution.model
                else len(prefs) + 1
            )
            for rank, name in enumerate(prefs):
                if rank >= current:
                    break
                if _canonical(name) in installed or not self._fits(name, None, hardware):
                    continue
                reason = (
                    "pull it; re-index required (embeddings are model-specific)"
                    if role == ROLE_EMBED
                    else "better option available"
                )
                out.append(Recommendation(role, name, reason))
                break
        return out

    def report(self, hardware: Hardware | None = None) -> Report:
        hw = hardware or self._probe()
        resolutions = self.resolve_all(hw)
        selected: dict[str, list[str]] = {}
        for role, resolution in resolutions.items():
            if resolution.model:
                selected.setdefault(_canonical(resolution.model), []).append(role)
        preferred = {_canonical(n) for names in self._catalog.roles.values() for n in names}
        rows: list[ModelRow] = []
        for info in self._installed:
            key = _canonical(info.name)
            roles = tuple(selected.get(key, ()))
            flags: list[Flag] = []
            warning = self._catalog.warnings.get(key) or self._catalog.warnings.get(info.name)
            if warning:
                flags.append(Flag(FLAG_WARNING, warning))
            if not roles and key not in preferred:
                size = f"{(info.size_bytes or 0) / (1024**3):.1f} GB"
                flags.append(Flag(FLAG_UNUSED, f"{size}, unused; consider removing"))
            if not flags:
                flags.append(Flag(FLAG_OK, "ok"))
            rows.append(ModelRow(info, roles, tuple(flags)))
        return Report(hw, resolutions, rows, self.recommendations(resolutions, hw))

    # ------------------------------------------------------------------ embed pinning
    def reindex_notice(self, new_model: str, indexed_files: int) -> ReindexNotice | None:
        """What switching the pinned embedder costs; ``None`` if it is not a change."""
        current = self._pinned_embed
        if current is not None and _canonical(current) == _canonical(new_model):
            return None
        seconds = indexed_files * _SECONDS_PER_FILE
        return ReindexNotice(
            files=indexed_files,
            estimated_seconds=seconds,
            message=(
                f"Switching embedder to {new_model} requires re-indexing "
                f"~{indexed_files} files (about {seconds / 60:.0f} min)."
            ),
        )
