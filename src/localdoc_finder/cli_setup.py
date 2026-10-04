"""``ldf setup``: install Ollama if needed, pick and download models, speed-test, save settings.

A thin front-end over ``SetupFlow``; the wizard in the app drives the same flow.
"""

import argparse
import sys
from collections.abc import Callable

from localdoc_finder.core.features import FEATURES, enabled_features
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.setup.flow import (
    SetupEvent,
    SetupOptions,
    SetupPreview,
    SetupResult,
    Stage,
)
from localdoc_finder.core.setup.ollama_install import OllamaState
from localdoc_finder.core.setup.plan import EXTRA_ROLES, SetupChoices
from localdoc_finder.core.setup.wiring import FlowBuilder, build_flow
from localdoc_finder.core.store.sqlite import StateDb

Printer = Callable[[str], None]
Asker = Callable[[str], bool]
PROGRESS_STEP = 0.1  # print a download line at most every 10 %


def add_setup_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    setup = sub.add_parser("setup", help="install Ollama, pick and download models, speed-test")
    setup.add_argument("--yes", action="store_true", help="accept every prompt")
    setup.add_argument("--embed", metavar="MODEL", help="embedding model (default: auto-picked)")
    setup.add_argument("--chat", metavar="MODEL", help="chat model (default: auto-picked)")
    setup.add_argument(
        "--extras", nargs="*", default=[], choices=EXTRA_ROLES, help="also download these roles"
    )
    setup.add_argument(
        "--features",
        nargs="*",
        choices=FEATURES,
        help="optional features to set up (default: keep the ones already on; search is always on)",
    )
    setup.add_argument("--dry-run", action="store_true", help="show the plan, change nothing")
    setup.add_argument(
        "--no-install-ollama", action="store_true", help="never download or install Ollama"
    )
    setup.add_argument("--skip-bench", action="store_true", help="skip the speed test")


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
    features = enabled_features(settings) if args.features is None else frozenset(args.features)
    choices = SetupChoices(
        args.embed, args.chat, tuple(args.extras), tuple(f for f in FEATURES if f in features)
    )
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
