"""Evaluation harness: compare embedding models on YOUR queries (recall@10, MRR, speed).

Each model gets its own throw-away index under ``<data dir>/eval/`` so the real index is never
touched. A query is a hit when any ``expect`` substring appears in a result path (case-insensitive,
``/`` and ``\\`` are equivalent); results are collapsed to one entry per file.
"""

import logging
import re
import shutil
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from vector_embed.core.doctypes.base import DocTypeClassifierSet
from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.indexer import Indexer
from vector_embed.core.power import PowerGate
from vector_embed.core.projects import Projects
from vector_embed.core.providers.base import EmbedKind
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import PowerSettings, Settings
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.search import SearchSkill
from vector_embed.core.store.lance import LanceStore
from vector_embed.core.store.sqlite import QueueItem, StateDb

logger = logging.getLogger(__name__)

K = 10
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class EvaluationError(ValueError):
    """The evaluation spec is invalid."""


@dataclass(frozen=True)
class EvalQuery:
    text: str
    expect: tuple[str, ...]


@dataclass(frozen=True)
class EvalSpec:
    corpus: tuple[Path, ...]
    models: tuple[str, ...]
    queries: tuple[EvalQuery, ...]


@dataclass
class ModelResult:
    model: str
    recall: float = 0.0
    mrr: float = 0.0
    index_seconds: float = 0.0
    p50_ms: float = 0.0
    files: int = 0
    missed: list[str] = field(default_factory=list)
    error: str = ""


class EvalEmbedder(Protocol):
    """What the harness needs from an embedder."""

    @property
    def dim(self) -> int: ...

    def embed(
        self,
        texts: list[str],
        kind: EmbedKind = "doc",
        cpu: bool = False,
        stop_check: Callable[[], bool] | None = None,
    ) -> np.ndarray: ...


EmbedderFactory = Callable[[str], EvalEmbedder]


def load_spec(path: Path) -> EvalSpec:
    """Read ``queries.yaml``: ``corpus`` directories, ``models`` and ``queries`` (q + expect)."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvaluationError("the evaluation file must be a mapping")
    queries = []
    for item in raw.get("queries") or []:
        expect = item.get("expect") or []
        if not item.get("q") or not expect:
            raise EvaluationError(f"every query needs 'q' and 'expect': {item!r}")
        queries.append(EvalQuery(str(item["q"]), tuple(str(e) for e in expect)))
    if not queries:
        raise EvaluationError("no queries in the evaluation file")
    base = path.parent
    corpus = tuple(
        (base / str(c)).resolve() if not Path(str(c)).is_absolute() else Path(str(c))
        for c in raw.get("corpus") or []
    )
    if not corpus:
        raise EvaluationError("no corpus directories in the evaluation file")
    models = tuple(str(m) for m in raw.get("models") or [])
    return EvalSpec(corpus, models, tuple(queries))


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def rank_of(paths: Sequence[str], expect: Sequence[str]) -> int:
    """1-based rank of the first result matching any expected substring, else 0."""
    wanted = [_norm(e) for e in expect]
    for rank, path in enumerate(paths[:K], 1):
        if any(w in _norm(path) for w in wanted):
            return rank
    return 0


class Evaluator:
    def __init__(
        self,
        settings: Settings,
        scope: ScopePolicy,
        embedder_factory: EmbedderFactory,
        workdir: Path,
        exclude: Sequence[Path] = (),
    ) -> None:
        self._settings = settings
        self._scope = scope
        self._factory = embedder_factory
        self._workdir = workdir
        self._exclude = {_norm(str(p)) for p in exclude}  # the answer key must not be indexed

    def evaluate(self, model: str, spec: EvalSpec, reuse: bool = False) -> ModelResult:
        result = ModelResult(model)
        try:
            embedder = self._factory(model)
            data_dir = self._workdir / _SAFE.sub("_", model)
            if data_dir.exists() and not reuse:
                shutil.rmtree(data_dir, ignore_errors=True)
            with StateDb(data_dir) as state:
                store = LanceStore(data_dir, state, model, dim=embedder.dim)
                if store.count() == 0:
                    result.files, result.index_seconds = self._index(
                        spec, state, store, embedder, model
                    )
                store.maintain()
                self._score(spec, state, store, embedder, result)
        except Exception as exc:  # one broken model must not abort the comparison
            logger.warning("eval: %s failed", model, exc_info=True)
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def _index(
        self,
        spec: EvalSpec,
        state: StateDb,
        store: LanceStore,
        embedder: EvalEmbedder,
        model: str,
    ) -> tuple[int, float]:
        settings = self._settings.model_copy(
            update={"embedding": self._settings.embedding.model_copy(update={"model": model})}
        )
        projects = Projects(self._scope, settings.scope, roots=spec.corpus)
        extractors = ExtractorSet(
            ExtractContext(self._scope, settings.scope, chunking=settings.chunking)
        )
        indexer = Indexer(
            settings,
            state,
            store,
            embedder,
            extractors=extractors,
            projects=projects,
            scope=self._scope,
            classifier=DocTypeClassifierSet(settings.doctypes, settings.scope),
        )
        paths = [
            str(p) for root in spec.corpus for p in projects.iter_files([root])
            if _norm(str(p)) not in self._exclude
        ]  # fmt: skip
        started = time.perf_counter()
        batch = settings.chunking.worker_batch_files
        for start in range(0, len(paths), batch):
            indexer.process([QueueItem(p, "upsert", 0) for p in paths[start : start + batch]])
        return len(paths), time.perf_counter() - started

    def _score(
        self,
        spec: EvalSpec,
        state: StateDb,
        store: LanceStore,
        embedder: EvalEmbedder,
        result: ModelResult,
    ) -> None:
        power = PowerGate(PowerSettings(), probe=lambda: True)
        power.update()
        skill = SearchSkill(SkillContext(self._settings, state, store, embedder, power))
        ranks: list[int] = []
        timings: list[float] = []
        for query in spec.queries:
            started = time.perf_counter()
            found = skill.search(query.text, limit=K)
            timings.append((time.perf_counter() - started) * 1000)
            rank = rank_of([r.path for r in found], query.expect)
            ranks.append(rank)
            if rank == 0:
                result.missed.append(query.text)
        hits = [r for r in ranks if r]
        result.recall = len(hits) / len(ranks)
        result.mrr = sum(1 / r for r in hits) / len(ranks)
        result.p50_ms = statistics.median(timings)


def format_results(results: Sequence[ModelResult]) -> str:
    """The comparison table used to choose the final models."""
    header = f"{'model':<28} {'recall@10':>9} {'MRR':>6} {'files':>6} {'index s':>8} {'p50 ms':>7}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: (r.error != "", -r.mrr)):
        if r.error:
            lines.append(f"{r.model:<28} FAILED: {r.error}")
            continue
        lines.append(
            f"{r.model:<28} {r.recall:>9.2f} {r.mrr:>6.3f} {r.files:>6} "
            f"{r.index_seconds:>8.1f} {r.p50_ms:>7.0f}"
        )
        if r.missed:
            lines.append("    missed: " + "; ".join(r.missed[:5]))
    return "\n".join(lines)
