"""``ve setup``: install Ollama if needed, pick and download models, speed-test, save settings.

A thin front-end over ``SetupFlow``; the wizard in the app drives the same flow.
"""

import argparse
import sys
from collections.abc import Callable
from typing import Protocol

from vector_embed.core import runtime
from vector_embed.core.models.benchmark import BenchResult, bench_chat, bench_embed
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import probe_hardware
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings import SETTINGS_FILENAME, Settings
from vector_embed.core.setup.flow import (
    SetupEvent,
    SetupFlow,
    SetupOptions,
    SetupPreview,
    SetupResult,
    SlowOffer,
    Stage,
)
from vector_embed.core.setup.ollama_install import OllamaSetup, OllamaState
from vector_embed.core.setup.plan import EXTRA_ROLES, SetupChoices
from vector_embed.core.setup.windows_ollama import WindowsOllamaSystem
from vector_embed.core.store.sqlite import StateDb

Printer = Callable[[str], None]
Asker = Callable[[str], bool]
PROGRESS_STEP = 0.1  # print a download line at most every 10 %


class FlowBuilder(Protocol):
    def __call__(
        self,
        settings: Settings,
        state: StateDb,
        progress: Callable[[SetupEvent], None],
        accept_downgrade: Callable[[SlowOffer], bool],
    ) -> SetupFlow: ...


def add_setup_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    setup = sub.add_parser("setup", help="install Ollama, pick and download models, speed-test")
    setup.add_argument("--yes", action="store_true", help="accept every prompt")
    setup.add_argument("--embed", metavar="MODEL", help="embedding model (default: auto-picked)")
    setup.add_argument("--chat", metavar="MODEL", help="chat model (default: auto-picked)")
    setup.add_argument(
        "--extras", nargs="*", default=[], choices=EXTRA_ROLES, help="also download these roles"
    )
    setup.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    setup.add_argument(
        "--no-install-ollama", action="store_true", help="never download or install Ollama"
    )
    setup.add_argument("--skip-bench", action="store_true", help="skip the speed test")


def build_flow(
    settings: Settings,
    state: StateDb,
    progress: Callable[[SetupEvent], None],
    accept_downgrade: Callable[[SlowOffer], bool],
) -> SetupFlow:
    provider = runtime.build_provider(settings)

    def embed_bench(model: str) -> BenchResult:
        tuned = settings.embedding.model_copy(update={"model": model})
        return bench_embed(OllamaProvider(tuned, client=provider.client))

    return SetupFlow(
        ollama=OllamaSetup(WindowsOllamaSystem(settings.ollama_host)),
        host=provider,
        catalog=load_catalog(settings.storage.data_dir),
        hardware=probe_hardware(),
        state=state,
        settings_path=settings.storage.data_dir / SETTINGS_FILENAME,
        data_dir=settings.storage.data_dir,
        current_embed=settings.embedding.model,
        bench_chat=lambda model: bench_chat(provider, model),
        bench_embed=embed_bench,
        progress=progress,
        accept_downgrade=accept_downgrade,
    )


def ask_yes_no(question: str) -> bool:
    sys.stderr.write(f"{question} [y/N] ")
    sys.stderr.flush()
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def format_preview(preview: SetupPreview) -> str:
    lines = [f"Ollama: {preview.ollama.value.replace('_', ' ')}"]
    for planned in preview.plan.models:
        size = f"{planned.download_mb} MB" if planned.download_mb else "size unknown"
        lines.append(f"  {planned.role}: {planned.model} ({size}) - {planned.reason}")
    lines.extend(f"  warning: {warning}" for warning in preview.plan.warnings)
    lines.append(
        f"To download: {preview.download_mb} MB; free disk: {preview.free_disk_mb} MB"
        + ("" if preview.enough_disk else " (NOT ENOUGH)")
    )
    return "\n".join(lines)


def format_result(result: SetupResult) -> str:
    lines = [
        f"Embedding model: {result.embed_model or '-'}",
        f"Chat model: {result.chat_model or '-'}",
    ]
    lines.extend(
        f"  {r.model}: {r.rate:.1f} {r.unit}, ready in {r.startup_s:.1f}s" for r in result.bench
    )
    lines.extend(f"  warning: {warning}" for warning in result.warnings)
    return "\n".join(lines)


class _ProgressPrinter:
    """One line per event, but a download only every ``PROGRESS_STEP``."""

    def __init__(self, write: Printer) -> None:
        self._write = write
        self._last: tuple[Stage, str, int] | None = None

    def __call__(self, event: SetupEvent) -> None:
        bucket = int((event.fraction or 0.0) / PROGRESS_STEP)
        key = (event.stage, event.message.split(":")[0], bucket)
        if key == self._last:
            return
        self._last = key
        suffix = f" {event.fraction:.0%}" if event.fraction is not None else ""
        self._write(f"[{event.stage.value}] {event.message}{suffix}")


def run_setup(
    args: argparse.Namespace,
    settings: Settings,
    out: Printer,
    err: Printer,
    *,
    ask: Asker = ask_yes_no,
    build: FlowBuilder = build_flow,
) -> int:
    choices = SetupChoices(args.embed, args.chat, tuple(args.extras))
    with StateDb(settings.storage.data_dir) as state:
        confirm: Asker = (lambda _question: True) if args.yes else ask
        flow = build(
            settings,
            state,
            _ProgressPrinter(err),
            lambda offer: confirm(
                f"{offer.model} runs at {offer.result.rate:.1f} {offer.result.unit}. "
                f"Switch to the smaller {offer.alternative}?"
            ),
        )
        preview = flow.preview(SetupOptions(choices))
        out(format_preview(preview))
        if args.dry_run:
            return 0
        install = _consent_to_install(args, preview, confirm)
        if not preview.enough_disk:
            err("Not enough disk space for the models.")
            return 1
        if (
            preview.to_download
            and not args.yes
            and not ask(f"Download {preview.download_mb} MB and continue?")
        ):
            err("Cancelled.")
            return 1
        result = flow.run(
            SetupOptions(choices, install_ollama=install, run_bench=not args.skip_bench)
        )
    out(format_result(result))
    return 0


def _consent_to_install(args: argparse.Namespace, preview: SetupPreview, confirm: Asker) -> bool:
    """Installing Ollama needs a yes, and ``--no-install-ollama`` is an unconditional no."""
    if preview.ollama is not OllamaState.MISSING or args.no_install_ollama:
        return False
    return confirm("Ollama is not installed. Download and install it from ollama.com?")
