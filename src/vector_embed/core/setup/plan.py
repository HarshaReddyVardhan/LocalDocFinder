"""The pure part of setup: which models to download for this machine and the user's choices."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from vector_embed.core.features import FEATURES
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_EMBED, ROLES, Catalog
from vector_embed.core.models.fit import budget_mb, fits
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.starter import StarterPlan, pick_starter

DISK_HEADROOM_MB = 2048  # free space to keep after the downloads (index, temp files)
EXTRA_ROLES = tuple(role for role in ROLES if role not in (ROLE_EMBED, ROLE_CHAT))


class SetupPlanError(ValueError):
    """The requested setup is not possible (unknown role, locked embedder)."""


@dataclass(frozen=True)
class SetupChoices:
    """What the user asked for; ``None`` means "pick for my hardware"."""

    embed: str | None = None
    chat: str | None = None
    extras: tuple[str, ...] = ()  # extra roles (caption, reranker, ...) to download a model for
    features: tuple[
        str, ...
    ] = ()  # optional features wanted (ask, chat, match); none = search only

    @property
    def wants_chat_model(self) -> bool:
        """Every feature answers with a chat model; search alone needs only the embedder."""
        return bool(self.features) or self.chat is not None


@dataclass(frozen=True)
class PlannedModel:
    role: str
    model: str
    download_mb: int  # 0 when unknown
    reason: str
    fits: bool
    downgrade: str | None = None  # smaller fitting model offered when the speed test says "slow"


@dataclass(frozen=True)
class SetupPlan:
    models: tuple[PlannedModel, ...]
    warnings: tuple[str, ...] = field(default=())

    def model_for(self, role: str) -> PlannedModel | None:
        return next((m for m in self.models if m.role == role), None)

    def to_download(self, installed: Collection[str]) -> tuple[str, ...]:
        """Distinct model names that are not installed yet, in role order."""
        have = {_canonical(name) for name in installed}
        return tuple(dict.fromkeys(m.model for m in self.models if _canonical(m.model) not in have))

    def download_mb(self, installed: Collection[str]) -> int:
        sizes = {m.model: m.download_mb for m in self.models}
        return sum(sizes[name] for name in self.to_download(installed))


def _canonical(name: str) -> str:
    return name.removesuffix(":latest")


def plan_setup(
    catalog: Catalog,
    hardware: Hardware,
    choices: SetupChoices = SetupChoices(),  # noqa: B008  # frozen dataclass, safe as a default
    *,
    locked_embed: str | None = None,
) -> SetupPlan:
    """Auto-pick for ``hardware``, honouring explicit choices.

    ``locked_embed`` is the embedder the existing index was built with; it is kept as is, because
    vectors from different models cannot be mixed (change it with ``ve models --embedder``).
    """
    unknown = sorted(set(choices.extras) - set(EXTRA_ROLES))
    if unknown:
        raise SetupPlanError(f"unknown extra roles: {', '.join(unknown)}")
    unknown_features = sorted(set(choices.features) - set(FEATURES))
    if unknown_features:
        raise SetupPlanError(f"unknown features: {', '.join(unknown_features)}")
    roles = (ROLE_EMBED, *((ROLE_CHAT,) if choices.wants_chat_model else ()), *choices.extras)
    starter = pick_starter(catalog, hardware, roles)
    warnings = [f"no {role} model fits this machine" for role in starter.missing_roles]
    wanted = {ROLE_EMBED: locked_embed or choices.embed, ROLE_CHAT: choices.chat}
    models: list[PlannedModel] = []
    for role in roles:
        planned = _planned(catalog, hardware, starter, role, wanted.get(role))
        if planned is None:
            continue
        models.append(planned)
        if not planned.fits:
            warnings.append(f"{planned.model} may not fit this machine")
    if locked_embed and choices.embed and choices.embed != locked_embed:
        warnings.append(
            f"an index already exists, so the embedder stays {locked_embed}; "
            "switch with `ve models --embedder` (it re-indexes)"
        )
    return SetupPlan(tuple(models), tuple(warnings))


def _planned(
    catalog: Catalog, hardware: Hardware, starter: StarterPlan, role: str, chosen: str | None
) -> PlannedModel | None:
    pick = starter.pick_for(role)
    if chosen is None:
        if pick is None:
            return None
        return PlannedModel(
            role, pick.model, pick.download_mb, pick.reason, fits=True, downgrade=pick.downgrade
        )
    entry = catalog.entry(chosen)
    needed = entry.vram_mb if entry else None
    ok = fits(entry, needed, hardware, budget_mb(hardware, total=True))
    reason = "chosen by you" if entry else "chosen by you (not in the catalog; size unknown)"
    downgrade = pick.downgrade if pick and pick.model == chosen else None
    return PlannedModel(role, chosen, entry.download_mb if entry else 0, reason, ok, downgrade)


def has_enough_disk(plan: SetupPlan, installed: Sequence[str], free_mb: int) -> bool:
    return plan.download_mb(installed) + DISK_HEADROOM_MB <= free_mb
