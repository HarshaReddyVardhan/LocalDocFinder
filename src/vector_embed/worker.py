"""Incremental indexing worker. Spawned by the watcher when idle and on AC; exits when done.

    python -m vector_embed.worker --now [--path D:\\Projects\\x] [--reconcile] [--allow-battery]

Power is checked before every batch (and between embedding batches). If the laptop is
unplugged the worker commits what is done, unloads the model (``keep_alive=0``) and exits.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vector_embed.core import runtime
from vector_embed.core.doctypes.base import DocTypeClassifierSet
from vector_embed.core.extractors.base import ExtractorSet
from vector_embed.core.idle import IdleGate
from vector_embed.core.indexer import BatchEmbedder, Indexer
from vector_embed.core.logging_setup import configure_logging
from vector_embed.core.power import PowerGate
from vector_embed.core.process import single_instance, stop_requested
from vector_embed.core.projects import Projects
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.providers.ollama import Interrupted
from vector_embed.core.reconcile import reconcile
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings, load_settings
from vector_embed.core.store.lance import IndexSchemaError, LanceStore, ModelMismatchError
from vector_embed.core.store.sqlite import CHAT_LOCK, EMBEDDER_APPROVED_KEY, StateDb

logger = logging.getLogger("worker")

EXIT_OK = 0
EXIT_PROVIDER = 2
EXIT_USAGE = 3
EXIT_MISMATCH = 4
LAST_RECONCILE_KEY = "last_reconcile"
_PROVIDER_RETRY_SECONDS = 600


class WorkerGate(Protocol):
    def worker_may_continue(
        self, allow_battery: bool = False, respect_activity: bool = True
    ) -> tuple[bool, str]: ...


@dataclass(frozen=True)
class WorkerOptions:
    now: bool = False  # skip the debounce and the "user is active" yield
    paths: tuple[str, ...] = ()  # scan and index these directories instead of the queue only
    reconcile: bool = False  # scan every watch root for missed changes first
    allow_battery: bool = False
    limit: int | None = None  # stop after N files (testing)


@dataclass
class WorkerParts:
    """Everything ``run_worker`` needs; built for real by ``main`` and faked in tests."""

    settings: Settings
    state: StateDb
    store: LanceStore
    embedder: BatchEmbedder
    extractors: ExtractorSet
    projects: Projects
    scope: ScopePolicy
    classifier: DocTypeClassifierSet
    gate: WorkerGate
    unload: Callable[[], None]


def run_worker(parts: WorkerParts, options: WorkerOptions) -> int:
    """Drain the queue. Returns an exit code (0 ok, 2 when the model server is unreachable)."""
    gate = parts.gate
    ok, why = gate.worker_may_continue(options.allow_battery, respect_activity=False)
    if not ok:
        logger.info("not starting: %s", why)
        return EXIT_OK

    stopped = {"why": ""}

    def stop_check() -> bool:
        if stopped["why"]:
            return True
        if stop_requested(parts.settings.storage.data_dir):
            stopped["why"] = "stop requested"
            return True
        proceed, reason = gate.worker_may_continue(
            options.allow_battery, respect_activity=not options.now
        )
        if not proceed:
            stopped["why"] = reason
        return not proceed

    state = parts.state
    indexer = Indexer(
        parts.settings,
        state,
        parts.store,
        parts.embedder,
        extractors=parts.extractors,
        projects=parts.projects,
        scope=parts.scope,
        classifier=parts.classifier,
        stop_check=stop_check,
    )
    started = time.time()
    try:
        stale = state.take_stale_paths()  # case-variant duplicates found by a schema migration
        if stale:
            parts.store.delete_paths(stale)
            logger.info("removed %d duplicate case-variant paths", len(stale))
        if options.paths or options.reconcile:
            result = reconcile(
                state, parts.projects, parts.scope, list(options.paths) or None, stop_check
            )
            if not options.paths and not result.interrupted:
                state.set_meta(LAST_RECONCILE_KEY, str(time.time()))
        code = _drain(parts, options, indexer, stop_check)
        if code != EXIT_OK:
            return code
        if stopped["why"]:
            # Maintenance is heavy disk and CPU work; the reason we stopped (battery, the user
            # is back) means it must wait for the next run.
            logger.info("stopping early (%s); progress committed, queue kept", stopped["why"])
            return EXIT_OK
        parts.store.maintain()
        if indexer.stats.files:
            indexer.assign_version_groups()
        logger.info("done in %.1fs", time.time() - started, extra={"stats": indexer.stats.__dict__})
        return EXIT_OK
    finally:
        parts.extractors.release_models()
        parts.unload()  # keep_alive=0 -> VRAM back to 0


def _drain(
    parts: WorkerParts,
    options: WorkerOptions,
    indexer: Indexer,
    stop_check: Callable[[], bool],
) -> int:
    state = parts.state
    batch = parts.settings.chunking.worker_batch_files
    while not stop_check():
        items = state.claim(batch, ignore_debounce=options.now)
        if not items:
            break
        try:
            finished = indexer.process(items)
        except Interrupted:
            break
        except ProviderError:
            logger.exception("embedding failed; is Ollama running?")
            for item in items:  # the files are fine: wait for the server without using attempts
                state.defer(item.path, delay=_PROVIDER_RETRY_SECONDS)
            return EXIT_PROVIDER
        except Exception:  # one poisonous batch must not be extracted again and again
            logger.exception("batch failed; marking its files failed")
            for item in items:
                state.fail(item.path)
            continue
        state.done(finished)
        logger.info(
            "progress", extra={"stats": indexer.stats.__dict__, "queued": state.queue_size()}
        )
        if options.limit and indexer.stats.files >= options.limit:
            break
    return EXIT_OK


def build_gate(settings: Settings, state: StateDb) -> IdleGate:
    """Power and idle checks; cheap, and built before anything touches the model server."""
    return IdleGate(
        PowerGate(settings.power),
        settings.idle,
        chat_active=lambda: state.lock_held(CHAT_LOCK),
    )


def build_parts(settings: Settings, state: StateDb, gate: WorkerGate) -> WorkerParts:
    """Wire the real collaborators. May use the model server, so call it after the gate passes."""
    scope = runtime.build_scope(settings)
    provider = runtime.build_provider(settings)
    try:
        store = runtime.open_store(settings, state, provider)
        return WorkerParts(
            settings=settings,
            state=state,
            store=store,
            embedder=provider,
            extractors=runtime.build_extractors(settings, scope, provider),
            projects=runtime.build_projects(settings, scope),
            scope=scope,
            classifier=DocTypeClassifierSet(
                settings.doctypes, settings.scope, runtime.load_prototypes(state, provider)
            ),
            gate=gate,
            unload=provider.unload_embedder,
        )
    except BaseException:
        provider.unload_embedder()  # a half-built worker must not leave the model on the GPU
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--now", action="store_true", help="skip debounce and user-activity yield")
    parser.add_argument("--path", action="append", help="scan + index this directory (repeatable)")
    parser.add_argument("--reconcile", action="store_true", help="scan roots for missed changes")
    parser.add_argument("--allow-battery", action="store_true", help="index even when unplugged")
    parser.add_argument("--limit", type=int, help="stop after N files (testing)")
    parser.add_argument("--model", help="embedding model override (needs --reindex to change it)")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="with --model: approve rebuilding the whole index for a different model",
    )
    parser.add_argument(
        "--data-dir", help="store location (default %%LOCALAPPDATA%%\\VectorEmbedData)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    overrides: dict[str, object] = {}
    if args.model and args.model != settings.embedding.model and not args.reindex:
        logger.error(
            "--model %s differs from the configured embedder (%s); every vector would be "
            "replaced. Add --reindex to rebuild the index, or change it in Settings.",
            args.model,
            settings.embedding.model,
        )
        return EXIT_USAGE
    if args.model:
        overrides["embedding"] = settings.embedding.model_copy(update={"model": args.model})
    if args.data_dir:
        overrides["storage"] = settings.storage.model_copy(update={"data_dir": Path(args.data_dir)})
    settings = settings.model_copy(update=overrides)
    configure_logging("worker", runtime.log_dir(settings), settings.log_level)
    with single_instance("worker", settings.storage.data_dir) as acquired:
        if not acquired:
            logger.info("another worker is running")
            return EXIT_OK
        state = StateDb(settings.storage.data_dir)
        try:
            if args.model and args.reindex:
                state.set_meta(EMBEDDER_APPROVED_KEY, args.model)  # the user asked for the rebuild
            return _run(settings, state, args)
        finally:
            state.close()


def _run(settings: Settings, state: StateDb, args: argparse.Namespace) -> int:
    gate = build_gate(settings, state)
    ok, why = gate.worker_may_continue(args.allow_battery, respect_activity=False)
    if not ok:  # decided before the model server is touched, so a refusal never loads the GPU
        logger.info("not starting: %s", why)
        return EXIT_OK
    try:
        parts = build_parts(settings, state, gate)
    except ProviderError:
        logger.exception("could not reach the model server; is Ollama running?")
        return EXIT_PROVIDER
    except (ModelMismatchError, IndexSchemaError) as exc:
        logger.error("%s", exc)  # the index is untouched; the message says what to do
        return EXIT_MISMATCH
    return run_worker(
        parts,
        WorkerOptions(
            now=args.now,
            paths=tuple(args.path or ()),
            reconcile=args.reconcile,
            allow_battery=args.allow_battery,
            limit=args.limit,
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
