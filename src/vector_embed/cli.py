"""Command line front-end: ``ve <skill> ...``, ``ve index``, ``ve models``, ``ve setup`` and more.

Skill commands are generated from the skill registry, so a new skill appears here without
editing this file.
"""

import argparse
import logging
import sys
import types
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic.fields import FieldInfo

from vector_embed import mcp_server, watcher, worker
from vector_embed.cli_cloud import CloudCommandError, add_cloud_parsers, run_cloud, run_keys
from vector_embed.cli_setup import add_setup_parser, run_setup
from vector_embed.core import runtime
from vector_embed.core.autostart import Autostart, AutostartError
from vector_embed.core.doctor import format_checks, run_doctor
from vector_embed.core.evaluation import (
    EvalSpec,
    EvaluationError,
    Evaluator,
    format_results,
    load_spec,
)
from vector_embed.core.extractors.ocr import WindowsOcr
from vector_embed.core.health import collect_health, format_health
from vector_embed.core.logging_setup import configure_logging
from vector_embed.core.models.hardware import probe_hardware
from vector_embed.core.models.manager import ModelChangeError, ModelManager
from vector_embed.core.models.report import format_report
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.secrets import KeyringStore, KeyStoreError
from vector_embed.core.settings import SETTINGS_FILENAME, Settings, SettingsError, load_settings
from vector_embed.core.setup.flow import SetupError
from vector_embed.core.setup.plan import SetupPlanError
from vector_embed.core.skills.base import Skill, load_skills
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger("cli")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


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
        if name == positional:
            parser.add_argument(name, nargs="+", help=help_text)
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
        if value is None:
            continue
        values[name] = " ".join(value) if name == positional else value
    return values


def run_skill(skill_cls: type[Skill], args: argparse.Namespace, settings: Settings) -> int:
    fields = dict(skill_cls.Input.model_fields)
    params = skill_cls.Input(**collect_input(args, fields, skill_cls.cli_positional))
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
        settings.storage.data_dir / SETTINGS_FILENAME,
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
    """Compare embedding models on your own queries (recall@10, MRR, speed)."""
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)
    models = tuple(args.models) if args.models else spec.models
    if not models:
        raise EvaluationError("name at least one model with --models or in the spec")
    if args.corpus:
        spec = EvalSpec(tuple(Path(c) for c in args.corpus), spec.models, spec.queries)

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
    )
    results = []
    for model in models:
        err(f"evaluating {model} ...")
        results.append(evaluator.evaluate(model, spec, reuse=args.reuse))
    out(format_results(results))
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
        out(format_health(collect_health(state, store, registry, provider.loaded_models)))
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
    parser = argparse.ArgumentParser(prog="ve", description="Local search + chat-with-documents.")
    parser.add_argument(
        "--data-dir", help="data directory (default %%LOCALAPPDATA%%\\VectorEmbedData)"
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
    sub.add_parser("index", help="run the indexing worker now (ve index --help for options)")
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
    add_cloud_parsers(sub)
    add_setup_parser(sub)
    autostart = sub.add_parser("autostart", help="start the watcher and tray app with Windows")
    autostart.add_argument("state", choices=("on", "off"))
    sub.add_parser("mcp", help="serve search, ask and match to MCP clients over stdio")
    sub.add_parser("doctor", help="check the environment and explain any problem")
    sub.add_parser("status", help="index and queue state")
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
            settings = settings.model_copy(update={"storage": storage})
        configure_logging("cli", None, "WARNING")
        return _dispatch(args, settings)
    except tuple(kind for kind, _, _ in _ERRORS) as exc:
        return _report(exc)


def _status(settings: Settings) -> int:
    out(watcher.status(settings))
    return EXIT_OK


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    handlers: dict[str, Callable[[], int]] = {
        "index": lambda: worker.main([]),  # reached only through ``ve --data-dir X index``
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
    }
    handler = handlers.get(args.command)
    return handler() if handler else run_skill(args.skill, args, settings)


if __name__ == "__main__":
    sys.exit(main())
