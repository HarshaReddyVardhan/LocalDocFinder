"""The resumable first-run setup: Ollama, models, disk check, download, speed test, settings.

Every step is idempotent (installed models are skipped and Ollama resumes partial pulls), so an
interrupted setup is finished by running it again. The flow reports progress through a callback
and takes every collaborator by injection; the wizard, ``ve setup`` and tests are thin wrappers.
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from vector_embed.core.features import FEATURES
from vector_embed.core.models.benchmark import (
    BenchmarkError,
    BenchResult,
    Verdict,
    judge,
    record_result,
)
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_EMBED, Catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.providers.base import ModelInfo, ProviderError, PullProgress
from vector_embed.core.settings_io import set_setting, set_settings
from vector_embed.core.setup.ollama_install import (
    INSTALLED_SIZE_MB,
    INSTALLER_SIZE_MB,
    OllamaSetup,
    OllamaSetupError,
    OllamaState,
)
from vector_embed.core.setup.plan import (
    DISK_HEADROOM_MB,
    SetupChoices,
    SetupPlan,
    has_enough_disk,
    plan_setup,
)
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

SETUP_COMPLETED_KEY = "setup_completed_at"
DOWNLOADS_DIRNAME = "downloads"


class SetupError(RuntimeError):
    """Setup cannot continue; the message tells the user what to do."""


class SetupCancelled(SetupError):  # noqa: N818  # not a failure: the user closed the wizard
    """Raised inside the flow when the UI asked it to stop."""


class Stage(StrEnum):
    OLLAMA = "ollama"
    PLAN = "plan"
    DISK = "disk"
    PULL = "pull"
    BENCH = "bench"
    SAVE = "save"
    DONE = "done"


@dataclass(frozen=True)
class SetupEvent:
    stage: Stage
    message: str
    fraction: float | None = None  # 0..1 within the stage when it is measurable


@dataclass(frozen=True)
class SetupOptions:
    choices: SetupChoices = SetupChoices()  # noqa: RUF009  # frozen, safe to share
    install_ollama: bool = False  # the user's consent to download and run the installer
    run_bench: bool = True


@dataclass(frozen=True)
class SlowOffer:
    """Shown when a model measured slow and a smaller one would fit."""

    role: str
    model: str
    result: BenchResult
    alternative: str


@dataclass(frozen=True)
class SetupPreview:
    """What ``run`` would do, computed without changing anything."""

    ollama: OllamaState
    plan: SetupPlan
    to_download: tuple[str, ...]
    download_mb: int
    free_disk_mb: int
    enough_disk: bool


@dataclass(frozen=True)
class EnvironmentProbe:
    """What the machine looks like right now (see ``SetupFlow.probe``)."""

    ollama: OllamaState
    installed: tuple[str, ...]
    free_disk_mb: int
    models: tuple[ModelInfo, ...] = ()  # the installed models in full, to offer the user's own


@dataclass(frozen=True)
class SetupResult:
    embed_model: str | None
    chat_model: str | None
    downloaded: tuple[str, ...]
    bench: tuple[BenchResult, ...]
    warnings: tuple[str, ...] = field(default=())


class ModelHost(Protocol):
    def list_models(self) -> list[ModelInfo]: ...
    def pull(self, model: str) -> Iterator[PullProgress]: ...


BenchFn = Callable[[str], BenchResult]
ProgressSink = Callable[[SetupEvent], None]


def _ignore(event: SetupEvent) -> None:
    return None


class SetupFlow:
    def __init__(  # collaborators are injected so every one can be faked
        self,
        *,
        ollama: OllamaSetup,
        host: ModelHost,
        catalog: Catalog,
        hardware: Hardware,
        state: StateDb,
        settings_path: Path,
        data_dir: Path,
        current_embed: str | None = None,
        bench_chat: BenchFn | None = None,
        bench_embed: BenchFn | None = None,
        progress: ProgressSink = _ignore,
        accept_downgrade: Callable[[SlowOffer], bool] = lambda offer: False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ollama = ollama
        self._host = host
        self._catalog = catalog
        self._hardware = hardware
        self._state = state
        self._settings_path = settings_path
        self._data_dir = data_dir
        self._current_embed = current_embed
        self._bench_fns = {ROLE_CHAT: bench_chat, ROLE_EMBED: bench_embed}
        self._progress = progress
        self._accept_downgrade = accept_downgrade
        self._clock = clock
        self._pulled: list[str] = []
        # Set by the UI: True once the user closed the wizard. Checked between and during steps.
        self.cancelled: Callable[[], bool] = lambda: False

    # ------------------------------------------------------------------ preview
    def _locked_embed(self) -> str | None:
        """The embedder an existing index depends on, if there is an index."""
        return self._current_embed if self._state.manifest_count() > 0 else None

    def probe(self) -> EnvironmentProbe:
        """Look at the machine: is Ollama up, what is installed, how much disk is free.

        This is the slow part of a preview (network and disk), so it is separate: a UI asks once,
        in the background, and then previews every choice against the same answer.
        """
        state = self._ollama.detect()
        models = tuple(self._host.list_models()) if state is OllamaState.RUNNING else ()
        return EnvironmentProbe(
            state, tuple(m.name for m in models), self._ollama.free_disk_mb(), models
        )

    def preview_from(self, probe: EnvironmentProbe, options: SetupOptions) -> SetupPreview:
        """What ``run`` would do given ``probe`` and the choices: pure arithmetic, instant."""
        plan = plan_setup(
            self._catalog, self._hardware, options.choices, locked_embed=self._locked_embed()
        )
        installed = list(probe.installed)
        return SetupPreview(
            ollama=probe.ollama,
            plan=plan,
            to_download=plan.to_download(installed),
            download_mb=plan.download_mb(installed),
            free_disk_mb=probe.free_disk_mb,
            enough_disk=has_enough_disk(plan, installed, probe.free_disk_mb),
        )

    def preview(self, options: SetupOptions) -> SetupPreview:
        return self.preview_from(self.probe(), options)

    # ------------------------------------------------------------------ run
    def run(self, options: SetupOptions) -> SetupResult:
        self._pulled = []
        self._checkpoint()
        self._ensure_ollama(options)
        self._checkpoint()
        installed = self._installed_names()
        plan = plan_setup(
            self._catalog, self._hardware, options.choices, locked_embed=self._locked_embed()
        )
        self._emit(Stage.PLAN, ", ".join(f"{m.role}: {m.model}" for m in plan.models) or "nothing")
        self._check_disk(plan, installed)
        self._pull_missing(plan.to_download(installed))

        chosen = {m.role: m.model for m in plan.models}
        results: list[BenchResult] = []
        warnings = list(plan.warnings)
        if options.run_bench:
            chosen, results = self._speed_test(plan, chosen, warnings)
        self._save(chosen, options.choices.features)
        self._emit(Stage.DONE, "setup complete")
        return SetupResult(
            chosen.get(ROLE_EMBED),
            chosen.get(ROLE_CHAT),
            tuple(self._pulled),
            tuple(results),
            tuple(warnings),
        )

    def _emit(self, stage: Stage, message: str, fraction: float | None = None) -> None:
        self._progress(SetupEvent(stage, message, fraction))
        self._checkpoint()  # progress arrives often (every download chunk): a cheap place to stop

    def _checkpoint(self) -> None:
        if self.cancelled():
            raise SetupCancelled("Setup was cancelled.")

    def _installed_names(self) -> list[str]:
        return [m.name for m in self._host.list_models()]

    # ------------------------------------------------------------------ steps
    def _ensure_ollama(self, options: SetupOptions) -> None:
        state = self._ollama.detect()
        if state is OllamaState.RUNNING:
            self._emit(Stage.OLLAMA, "Ollama is running")
            return
        try:
            if state is OllamaState.INSTALLED_NOT_RUNNING:
                self._emit(Stage.OLLAMA, "starting Ollama")
                self._ollama.start_server()
                return
            if not options.install_ollama:
                raise SetupError(
                    "Ollama is not installed. Allow setup to install it, or install it from "
                    "https://ollama.com/download and run setup again."
                )
            self._check_install_disk()
            self._emit(Stage.OLLAMA, "downloading the Ollama installer")
            self._ollama.install(
                self._data_dir / DOWNLOADS_DIRNAME,
                lambda done, total: self._emit(
                    Stage.OLLAMA, "downloading Ollama", done / total if total else None
                ),
                consented=True,
                checkpoint=self._checkpoint,
            )
        except OllamaSetupError as exc:
            raise SetupError(str(exc)) from exc

    def _check_install_disk(self) -> None:
        """Room for the installer and for the program it unpacks, before downloading a byte."""
        downloads = self._data_dir / DOWNLOADS_DIRNAME
        need = INSTALLER_SIZE_MB + INSTALLED_SIZE_MB + DISK_HEADROOM_MB
        free = self._ollama.free_disk_mb_at(downloads)
        if free < need:
            raise SetupError(
                f"Not enough disk space to install Ollama: need about {need} MB "
                f"(the installer and the program), {free} MB free. Free some space and try again."
            )

    def _check_disk(self, plan: SetupPlan, installed: list[str]) -> None:
        free = self._ollama.free_disk_mb()
        if has_enough_disk(plan, installed, free):
            self._emit(Stage.DISK, f"{free} MB free")
            return
        need = plan.download_mb(installed) + DISK_HEADROOM_MB
        raise SetupError(
            f"Not enough disk space for the models: need about {need} MB, {free} MB free. "
            "Free some space or choose smaller models."
        )

    def _pull_missing(self, names: tuple[str, ...]) -> None:
        for index, name in enumerate(names, start=1):
            label = f"({index}/{len(names)}) {name}"
            self._emit(Stage.PULL, label, 0.0)
            for step in self._host.pull(name):
                self._emit(
                    Stage.PULL, f"{label}: {step.status}", step.fraction if step.total else None
                )
            self._pulled.append(name)

    def _speed_test(
        self, plan: SetupPlan, chosen: dict[str, str], warnings: list[str]
    ) -> tuple[dict[str, str], list[BenchResult]]:
        """Measure the embedder, then the chat model: one at a time, never together."""
        final = dict(chosen)
        results: list[BenchResult] = []
        for role in (ROLE_EMBED, ROLE_CHAT):
            planned = plan.model_for(role)
            bench = self._bench_fns[role]
            if planned is None or bench is None or self._locked(role):
                continue
            result = self._measure(role, planned.model, bench, warnings)
            if result is None:
                continue
            results.append(result)
            if judge(result) is Verdict.SLOW and planned.downgrade:
                smaller = self._try_downgrade(role, planned.downgrade, result, bench, warnings)
                if smaller is not None:
                    final[role] = smaller.model
                    results.append(smaller)
        return final, results

    def _locked(self, role: str) -> bool:
        return role == ROLE_EMBED and self._locked_embed() is not None

    def _measure(
        self, role: str, model: str, bench: BenchFn, warnings: list[str]
    ) -> BenchResult | None:
        self._emit(Stage.BENCH, f"speed test: {model}")
        try:
            result = bench(model)
        except (BenchmarkError, ProviderError) as exc:
            logger.warning("setup: speed test of %s failed: %s", model, exc)
            warnings.append(f"could not speed-test {model}: {exc}")
            return None
        record_result(self._state, result)
        self._emit(Stage.BENCH, f"{model}: {result.rate:.1f} {result.unit}")
        return result

    def _try_downgrade(
        self, role: str, alternative: str, slow: BenchResult, bench: BenchFn, warnings: list[str]
    ) -> BenchResult | None:
        offer = SlowOffer(role, slow.model, slow, alternative)
        if not self._accept_downgrade(offer):
            warnings.append(f"{slow.model} is slow on this machine ({slow.rate:.1f} {slow.unit})")
            return None
        self._pull_missing(tuple(n for n in (alternative,) if n not in self._installed_names()))
        return self._measure(role, alternative, bench, warnings)

    def _save(self, chosen: dict[str, str], features: tuple[str, ...]) -> None:
        self._emit(Stage.SAVE, "saving settings")
        set_settings(
            self._settings_path, [(["features", name], name in features) for name in FEATURES]
        )
        embed = chosen.get(ROLE_EMBED)
        if embed:
            set_setting(self._settings_path, ["embedding", "model"], embed)
        chat = chosen.get(ROLE_CHAT)
        if chat:
            set_setting(self._settings_path, ["models", "overrides", ROLE_CHAT], chat)
        self._state.set_meta(SETUP_COMPLETED_KEY, str(self._clock()))
