"""Command line front-end: ``ldf <skill> ...``, ``ldf index``, ``ldf models``, ``ldf setup``, ...

Skill commands are generated from the skill registry, so a new skill appears here without
editing this file.
"""

import argparse
import logging
import sys
import types
import typing
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic.fields import FieldInfo

from localdoc_finder import mcp_server, watcher, worker
from localdoc_finder.cli_cloud import CloudCommandError, add_cloud_parsers, run_cloud, run_keys
from localdoc_finder.cli_setup import add_setup_parser, run_setup
from localdoc_finder.core import runtime
from localdoc_finder.core.autostart import Autostart, AutostartError
from localdoc_finder.core.doctor import format_checks, run_doctor
from localdoc_finder.core.evaluation import (
    EvalSpec,
    EvaluationError,
    Evaluator,
    format_calibration,
    format_results,
    load_spec,
)
from localdoc_finder.core.extractors.ocr import WindowsOcr
from localdoc_finder.core.features import enabled_features, is_enabled
from localdoc_finder.core.health import collect_health, format_health
from localdoc_finder.core.logging_setup import configure_logging
from localdoc_finder.core.models.hardware import probe_hardware
from localdoc_finder.core.models.manager import ModelChangeError, ModelManager
from localdoc_finder.core.models.report import format_report
from localdoc_finder.core.ollama_service import ensure_ollama_running
from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.providers.ollama import OllamaProvider
from localdoc_finder.core.secrets import KeyringStore, KeyStoreError
from localdoc_finder.core.settings import Settings, SettingsError, load_settings
from localdoc_finder.core.setup.flow import SetupError
from localdoc_finder.core.setup.ollama_install import OllamaState
from localdoc_finder.core.setup.plan import SetupPlanError
from localdoc_finder.core.skills.base import Skill, load_skills
from localdoc_finder.core.store.sqlite import StateDb

logger = logging.getLogger("cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
DOCS_LIMIT = 50


def out(text: str = "") -> None:
    """The CLI's only stdout writer (library code never prints)."""
    sys.stdout.write(text + "\n")


def err(text: str) -> None:
    sys.stderr.write(text + "\n")


def _unwrap(annotation: object) -> object:
    """``int | None`` -> ``int``."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return args[0] if len(args) == 1 else annotation
    return annotation


def add_input_arguments(
    parser: argparse.ArgumentParser, fields: dict[str, FieldInfo], positional: str | None
) -> None:
    """Turn a skill's pydantic input model into argparse arguments."""
    for name, info in fields.items():
        help_text = info.description or ""
        if name == positional:  # optional when the field has a default (e.g. chat --list-...)
            parser.add_argument(name, nargs="+" if info.is_required() else "*", help=help_text)
            continue
        flag = "--" + name.replace("_", "-")
        kind = _unwrap(info.annotation)
        if kind is bool:
            parser.add_argument(
                flag, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_text
            )
        elif typing.get_origin(kind) is list:
            parser.add_argument(flag, dest=name, nargs="*", default=None, help=help_text)
        elif kind in (int, float, str):
            parser.add_argument(
                flag,
                dest=name,
                type=kind,
                default=None,
                required=info.is_required(),
                help=help_text,
            )
        else:
            raise ValueError(f"unsupported input field type for --{name}: {kind!r}")


def collect_input(
    args: argparse.Namespace, fields: dict[str, FieldInfo], positional: str | None
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in fields:
        value = getattr(args, name, None)
        if value is None or (name == positional and not value):
            continue
        values[name] = " ".join(value) if name == positional else value
    return values


def run_skill(skill_cls: type[Skill], args: argparse.Namespace, settings: Settings) -> int:
    if not is_enabled(skill_cls.name, enabled_features(settings)):
        err(
            f"ldf {skill_cls.name} is not enabled. Turn it on in Settings > Features, "
            f"or run `ldf setup --features {skill_cls.name}` to also download its model."
        )
        return EXIT_USAGE
    fields = dict(skill_cls.Input.model_fields)
    try:
        params = skill_cls.Input(**collect_input(args, fields, skill_cls.cli_positional))
    except ValidationError as exc:
        problem = exc.errors(include_input=False)[0]
        err(f"ldf {skill_cls.name}: {problem['msg'].removeprefix('Value error, ')}")
        return EXIT_USAGE
    with StateDb(settings.storage.data_dir) as state:
        ctx = runtime.build_skill_context(settings, state)
        if getattr(args, "cloud_ok", False):
            cloud = ctx.extras.get("cloud")
            if isinstance(cloud, runtime.CloudContext):
                cloud.consent.grant()  # the flag is the consent: this one command, nothing else
        skill = skill_cls(ctx)
        stream = skill.stream(params)
        if stream is not None:
            for delta in stream:
                sys.stdout.write(delta)
            sys.stdout.write("\n")
            return EXIT_OK
        out(skill.render(skill.run(params)))
    return EXIT_OK


def build_manager(settings: Settings, state: StateDb) -> ModelManager:
    provider = runtime.build_provider(settings)
    registry = runtime.build_model_registry(settings, state, provider)
    return ModelManager(
        registry,
        provider,
        state,
        settings.settings_path(),
        indexed_files=state.manifest_count,
    )


def cmd_models(settings: Settings, args: argparse.Namespace) -> int:
    with StateDb(settings.storage.data_dir) as state:
        manager = build_manager(settings, state)
        if args.pull:
            manager.pull(
                args.pull, lambda p: err(f"{p.status} {p.fraction:.0%}" if p.total else p.status)
            )
            out(f"pulled {args.pull}")
        for assignment in args.set or []:
            role, _, model = assignment.partition("=")
            manager.set_override(role, model or None)
            out(f"{role} -> {model or 'automatic'}")
        if args.embedder:
            notice = manager.change_embedder(args.embedder, confirmed=args.yes)
            out(notice.message if notice else "already the embedder")
        manager.registry.refresh()
        out(format_report(manager.registry.report()))
    return EXIT_OK


def cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    """Compare embedding models on your own queries (recall, precision, speed)."""
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)
    models = tuple(args.models) if args.models else spec.models
    if not models:
        raise EvaluationError("name at least one model with --models or in the spec")
    if args.corpus:
        spec = EvalSpec(tuple(Path(c) for c in args.corpus), spec.models, spec.queries)
    # The corpus was picked on purpose: index every kind of file in it, whatever the user's
    # file-type preset. Secret and blocked-folder rules still apply.
    settings = settings.model_copy(
        update={"scope": settings.scope.model_copy(update={"file_types": "everything"})}
    )

    def factory(model: str) -> OllamaProvider:
        return OllamaProvider(
            settings.embedding.model_copy(update={"model": model}), host=settings.ollama_host
        )

    evaluator = Evaluator(
        settings,
        runtime.build_scope(settings),
        factory,
        settings.storage.data_dir / "eval",
        exclude=[spec_path],
        ocr=WindowsOcr(settings.images.ocr_max_dimension),
    )
    results = []
    for model in models:
        err(f"evaluating {model} ...")
        results.append(evaluator.evaluate(model, spec, reuse=args.reuse))
    out(format_results(results))
    if args.calibrate:
        out("")
        out(format_calibration(results))
    return EXIT_OK if all(not r.error for r in results) else EXIT_ERROR


def cmd_mcp(settings: Settings) -> int:
    """stdout belongs to the MCP protocol, so logs go to stderr and the log file only."""
    configure_logging("mcp", runtime.log_dir(settings), "INFO")
    mcp_server.serve(settings)
    return EXIT_OK


def cmd_autostart(args: argparse.Namespace) -> int:
    enabled = args.state == "on"
    Autostart().apply(enabled)
    out("startup tasks registered" if enabled else "startup tasks removed")
    return EXIT_OK


def cmd_health(settings: Settings) -> int:
    provider = runtime.build_provider(settings)
    with StateDb(settings.storage.data_dir) as state:
        registry = runtime.build_model_registry(settings, state, provider)
        registry.refresh()
        store = runtime.open_read_only_store(settings, state)
        budget = settings.cloud.monthly_budget_usd
        snapshot = collect_health(state, store, registry, provider.loaded_models, budget_usd=budget)
        out(format_health(snapshot))
    return EXIT_OK


def cmd_doctor(settings: Settings) -> int:
    provider = runtime.build_provider(settings)
    with StateDb(settings.storage.data_dir) as state:
        registry = runtime.build_model_registry(settings, state, provider)
        checks = run_doctor(
            settings,
            registry,
            probe_hardware(),
            WindowsOcr(settings.images.ocr_max_dimension),
            state=state,
        )
    out(format_checks(checks))
    return EXIT_OK if all(c.ok for c in checks) else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ldf", description="Local search + chat-with-documents.")
    parser.add_argument(
        "--data-dir", help="data directory (default %%LOCALAPPDATA%%\\LocalDocFinderData)"
    )
    parser.add_argument(
        "--cloud-ok",
        action="store_true",
        help="allow this command to send (masked) text to the configured cloud provider",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for skill_cls in load_skills():
        command = sub.add_parser(skill_cls.name, help=skill_cls.description)
        add_input_arguments(command, dict(skill_cls.Input.model_fields), skill_cls.cli_positional)
        command.set_defaults(skill=skill_cls)
    sub.add_parser("index", help="run the indexing worker now (ldf index --help for options)")
    models = sub.add_parser("models", help="installed models, role choices and recommendations")
    models.add_argument(
        "--pull", metavar="MODEL", help="download a model (e.g. the recommended one)"
    )
    models.add_argument(
        "--set", action="append", metavar="ROLE=MODEL", help="pin a role (empty clears)"
    )
    models.add_argument("--embedder", metavar="MODEL", help="switch the embedder (re-index needed)")
    models.add_argument("--yes", action="store_true", help="confirm the re-index for --embedder")
    sub.add_parser("health", help="queue, VRAM, loaded models and cloud spend")
    evaluate = sub.add_parser("eval", help="compare embedding models on your own queries")
    evaluate.add_argument("--spec", default="eval/queries.yaml", help="queries file")
    evaluate.add_argument("--models", nargs="+", help="models to compare (default: the spec's)")
    evaluate.add_argument("--corpus", nargs="+", help="directories to index instead of the spec's")
    evaluate.add_argument("--reuse", action="store_true", help="reuse indexes from a previous run")
    evaluate.add_argument(
        "--calibrate",
        action="store_true",
        help="print cosine-similarity percentiles of true hits vs. negatives",
    )
    add_cloud_parsers(sub)
    add_setup_parser(sub)
    autostart = sub.add_parser("autostart", help="start the watcher and tray app with Windows")
    autostart.add_argument("state", choices=("on", "off"))
    sub.add_parser("mcp", help="serve search, ask and match to MCP clients over stdio")
    sub.add_parser("doctor", help="check the environment and explain any problem")
    sub.add_parser("status", help="index and queue state")
    docs = sub.add_parser("docs", help="list the files that are indexed (searchable)")
    docs.add_argument("contains", nargs="?", default="", help="only paths containing this text")
    docs.add_argument("--failed", action="store_true", help="only files that could not be indexed")
    docs.add_argument("--limit", type=int, default=DOCS_LIMIT, help="how many to show")
    return parser


# Order matters: subclasses (ModelChangeError, ProviderError) before their base RuntimeError.
_ERRORS: tuple[tuple[type[Exception], int, str], ...] = (
    (SettingsError, EXIT_USAGE, "settings error: {}"),
    (ProviderError, EXIT_ERROR, "model provider error: {}"),
    (ModelChangeError, EXIT_USAGE, "{}"),
    (CloudCommandError, EXIT_USAGE, "{}"),
    (EvaluationError, EXIT_USAGE, "{}"),
    (AutostartError, EXIT_ERROR, "{}"),
    (SetupPlanError, EXIT_USAGE, "{}"),
    (SetupError, EXIT_ERROR, "{}"),
    (KeyStoreError, EXIT_ERROR, "{}"),
    (RuntimeError, EXIT_ERROR, "{}"),  # skill-level errors, e.g. search disabled on battery
)


def _report(exc: Exception) -> int:
    for kind, code, template in _ERRORS:
        if isinstance(exc, kind):
            err(template.format(exc))
            return code
    raise exc


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "index":  # the worker has its own options; hand them over untouched
        return worker.main(tokens[1:])
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        if args.data_dir:
            storage = settings.storage.model_copy(update={"data_dir": Path(args.data_dir)})
            # A different data folder carries its own settings file.
            settings = settings.model_copy(update={"storage": storage, "settings_file": None})
        configure_logging("cli", None, "WARNING")
        return _dispatch(args, settings)
    except tuple(kind for kind, _, _ in _ERRORS) as exc:
        return _report(exc)


def cmd_docs(settings: Settings, args: argparse.Namespace) -> int:
    """Which files are in the index: newest first, with a count of the rest."""
    with StateDb(settings.storage.data_dir) as state:
        files = state.manifest_list(args.contains, failed_only=args.failed, limit=args.limit)
        total, queued = state.manifest_count(), state.queue_size()
    for entry in files:
        stamp = datetime.fromtimestamp(entry.indexed_at).strftime("%Y-%m-%d %H:%M")
        mark = "FAILED " if entry.failed else ""
        out(f"{stamp}  {entry.size // 1024:>8} KB  {mark}{entry.path}")
    out("")
    out(f"shown: {len(files)}; indexed in total: {total}; waiting in the queue: {queued}")
    return EXIT_OK


def _status(settings: Settings) -> int:
    out(watcher.status(settings))
    return EXIT_OK


_NO_MODEL_SERVER = frozenset({"keys", "cloud", "autostart", "doctor", "status", "docs", "setup"})


def _ensure_model_server(args: argparse.Namespace, settings: Settings) -> None:
    """Start Ollama when a command needs models and it is installed but stopped."""
    if args.command in _NO_MODEL_SERVER:
        return
    if ensure_ollama_running(settings.ollama_host) is not OllamaState.RUNNING:
        err("warning: Ollama is not running and could not be started; model calls will fail.")


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    _ensure_model_server(args, settings)
    handlers: dict[str, Callable[[], int]] = {
        "index": lambda: worker.main([]),  # reached only through ``ldf --data-dir X index``
        "models": lambda: cmd_models(settings, args),
        "health": lambda: cmd_health(settings),
        "eval": lambda: cmd_eval(settings, args),
        "mcp": lambda: cmd_mcp(settings),
        "keys": lambda: run_keys(args, settings, KeyringStore(), out),
        "cloud": lambda: run_cloud(args, settings, KeyringStore(), out),
        "setup": lambda: run_setup(args, settings, out, err),
        "autostart": lambda: cmd_autostart(args),
        "doctor": lambda: cmd_doctor(settings),
        "status": lambda: _status(settings),
        "docs": lambda: cmd_docs(settings, args),
    }
    handler = handlers.get(args.command)
    return handler() if handler else run_skill(args.skill, args, settings)


if __name__ == "__main__":
    sys.exit(main())
