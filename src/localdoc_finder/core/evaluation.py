"""Evaluation harness: compare embedding models on YOUR queries, for recall and for precision.

Each model gets its own throw-away index under ``<data dir>/eval/`` so the real index is never
touched. A result is relevant when any ``expect`` substring appears in its path (case-insensitive,
``/`` and ``\\`` are equivalent); results are collapsed to one entry per file.

Precision needs queries that should *not* match: ``expect_none: true`` marks a question the
corpus cannot answer, and ``not: [...]`` lists paths that must stay out of the top 5. Every query
runs twice, through the full search pipeline and through the vector leg alone, so the embedder's
quality is not mixed up with keyword and filename effects. ``--calibrate`` reports the cosine
similarities of true hits against negatives, the input for a per-model similarity floor.
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

from localdoc_finder.core.doctypes.base import DocTypeClassifierSet
from localdoc_finder.core.extractors.base import ExtractContext, ExtractorSet, NullOcr, OcrEngine
from localdoc_finder.core.indexer import Indexer
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.projects import Projects
from localdoc_finder.core.providers.base import EmbedKind
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.settings import PowerSettings, Settings
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.skills.search import SearchSkill
from localdoc_finder.core.store.lance import CHUNKS, LanceStore
from localdoc_finder.core.store.sqlite import QueueItem, StateDb

logger = logging.getLogger(__name__)

K = 10  # recall@K and nDCG@K
TOP = 5  # P@TOP, and the window a negative must stay out of
PERCENTILES = (10, 25, 50, 75, 90)
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class EvaluationError(ValueError):
    """The evaluation spec is invalid."""


@dataclass(frozen=True)
class EvalQuery:
    text: str
    expect: tuple[str, ...] = ()
    not_expected: tuple[str, ...] = ()
    expect_none: bool = False  # nothing in the corpus answers it

    @property
    def positive(self) -> bool:
        return bool(self.expect)

    @property
    def negative(self) -> bool:
        return self.expect_none or bool(self.not_expected)


@dataclass(frozen=True)
class EvalSpec:
    corpus: tuple[Path, ...]
    models: tuple[str, ...]
    queries: tuple[EvalQuery, ...]


@dataclass
class LegScore:
    """Quality of one retrieval path. Recall, MRR, nDCG and P@5 are averaged over the queries
    with ``expect``; the false-positive rate over the queries with a negative constraint."""

    recall: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    p_at_5: float = 0.0
    fp_rate: float = 0.0
    missed: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)


@dataclass
class ModelResult:
    model: str
    full: LegScore = field(default_factory=LegScore)
    vector: LegScore = field(default_factory=LegScore)
    index_seconds: float = 0.0
    p50_ms: float = 0.0
    files: int = 0
    true_similarities: list[float] = field(default_factory=list)
    negative_similarities: list[float] = field(default_factory=list)
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


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EvaluationError(f"expected a list, got {value!r}")
    return tuple(str(v) for v in value)


def _query(item: object) -> EvalQuery:
    if not isinstance(item, dict) or not item.get("q"):
        raise EvaluationError(f"every query needs 'q': {item!r}")
    query = EvalQuery(
        str(item["q"]),
        _strings(item.get("expect")),
        _strings(item.get("not")),
        bool(item.get("expect_none", False)),
    )
    if query.positive == query.expect_none:
        raise EvaluationError(f"every query needs either 'expect' or 'expect_none': {item!r}")
    return query


def load_spec(path: Path) -> EvalSpec:
    """Read ``queries.yaml``: ``corpus`` directories, ``models`` and ``queries``.

    A query has ``q`` and either ``expect`` (relevant path substrings) or ``expect_none: true``,
    plus optional ``not`` (path substrings that must not reach the top 5).
    """
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise EvaluationError("the evaluation file must be a mapping")
    queries = tuple(_query(item) for item in raw.get("queries") or [])
    if not queries:
        raise EvaluationError("no queries in the evaluation file")
    base = path.parent
    corpus = tuple(
        (base / str(c)).resolve() if not Path(str(c)).is_absolute() else Path(str(c))
        for c in raw.get("corpus") or []
    )
    if not corpus:
        raise EvaluationError("no corpus directories in the evaluation file")
    return EvalSpec(corpus, _strings(raw.get("models")), queries)


# ---------------------------------------------------------------------- pure metrics
def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def _matches(path: str, patterns: Sequence[str]) -> bool:
    normed = _norm(path)
    return any(_norm(p) in normed for p in patterns)


def rank_of(paths: Sequence[str], expect: Sequence[str]) -> int:
    """1-based rank of the first result matching any expected substring, else 0."""
    for rank, path in enumerate(paths[:K], 1):
        if _matches(path, expect):
            return rank
    return 0


def ndcg_at(paths: Sequence[str], expect: Sequence[str], k: int = K) -> float:
    """Binary nDCG@k with one ideal hit per expected pattern.

    A pattern counts once, at the first result it matches, so a pattern matching two files
    cannot push the score above 1.
    """
    if not expect:
        return 0.0
    unmatched = [_norm(e) for e in expect]
    dcg = 0.0
    for rank, path in enumerate(paths[:k], 1):
        normed = _norm(path)
        hit = next((p for p in unmatched if p in normed), None)
        if hit is not None:
            unmatched.remove(hit)
            dcg += 1 / np.log2(rank + 1)
    ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(len(expect), k) + 1))
    return float(dcg / ideal)


def precision_at(paths: Sequence[str], expect: Sequence[str], k: int = TOP) -> float:
    """Share of the top ``k`` slots holding a relevant file."""
    return sum(_matches(p, expect) for p in paths[:k]) / k


def is_false_positive(
    paths: Sequence[str], query: EvalQuery, k: int = TOP, *, judge_none: bool = True
) -> bool:
    """A query the corpus cannot answer returned something, or a forbidden path reached the top."""
    top = paths[:k]
    if judge_none and query.expect_none and top:
        return True
    return any(_matches(p, query.not_expected) for p in top)


def score_leg(runs: Sequence[tuple[EvalQuery, Sequence[str]]], judge_none: bool = True) -> LegScore:
    """Aggregate per-query results. ``judge_none=False`` ignores ``expect_none`` when counting
    false positives: a bare nearest-neighbour search always returns something."""
    score = LegScore()
    positives = [(q, paths) for q, paths in runs if q.positive]
    for query, paths in positives:
        rank = rank_of(paths, query.expect)
        if rank:
            score.recall += 1
            score.mrr += 1 / rank
        else:
            score.missed.append(query.text)
        score.ndcg += ndcg_at(paths, query.expect)
        score.p_at_5 += precision_at(paths, query.expect)
    if positives:
        for name in ("recall", "mrr", "ndcg", "p_at_5"):
            setattr(score, name, getattr(score, name) / len(positives))
    judged = [(q, paths) for q, paths in runs if q.not_expected or (q.expect_none and judge_none)]
    for query, paths in judged:
        if is_false_positive(paths, query, judge_none=judge_none):
            score.false_positives.append(query.text)
    score.fp_rate = len(score.false_positives) / len(judged) if judged else 0.0
    return score


def percentiles(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    return [float(v) for v in np.percentile(np.asarray(values), PERCENTILES)]


# ---------------------------------------------------------------------- harness
@dataclass
class _VectorRun:
    paths: list[str]
    similarities: dict[str, float]  # best chunk per file


class Evaluator:
    def __init__(
        self,
        settings: Settings,
        scope: ScopePolicy,
        embedder_factory: EmbedderFactory,
        workdir: Path,
        *,
        exclude: Sequence[Path] = (),
        ocr: OcrEngine | None = None,
    ) -> None:
        self._settings = settings
        self._scope = scope
        self._factory = embedder_factory
        self._workdir = workdir
        self._exclude = {_norm(str(p)) for p in exclude}  # the answer key must not be indexed
        self._ocr = ocr or NullOcr()  # scanned pages and photos need it to have any text

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
            ExtractContext(
                self._scope,
                settings.scope,
                chunking=settings.chunking,
                images=settings.images,
                ocr=self._ocr,
            )
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
        full_runs: list[tuple[EvalQuery, Sequence[str]]] = []
        vector_runs: list[tuple[EvalQuery, Sequence[str]]] = []
        timings: list[float] = []
        for query in spec.queries:
            started = time.perf_counter()
            found = skill.search(query.text, limit=K)
            timings.append((time.perf_counter() - started) * 1000)
            full_runs.append((query, [r.path for r in found]))
            vector = self._vector_leg(store, embedder, query.text)
            vector_runs.append((query, vector.paths))
            self._collect_similarities(query, vector, result)
        result.full = score_leg(full_runs)
        result.vector = score_leg(vector_runs, judge_none=False)
        result.p50_ms = statistics.median(timings)

    def _vector_leg(self, store: LanceStore, embedder: EvalEmbedder, text: str) -> _VectorRun:
        vector = embedder.embed([text], kind="query")[0]
        cfg = self._settings.search
        rows = store.vector_search(
            CHUNKS, vector, ["path"], "", cfg.candidates, min_content_chars=cfg.min_content_chars
        )
        best: dict[str, float] = {}
        for row in rows:  # nearest first, so the first row of a file is its best chunk
            best.setdefault(row["path"], 1.0 - float(row["_distance"]))
        return _VectorRun(list(best), best)

    @staticmethod
    def _collect_similarities(query: EvalQuery, run: _VectorRun, result: ModelResult) -> None:
        """True hits: relevant files the vector leg found. Negatives: every other file in its top
        K, and everything returned for a query nothing should answer."""
        for path in run.paths[:K]:
            similarity = run.similarities[path]
            if query.positive and _matches(path, query.expect):
                result.true_similarities.append(similarity)
            else:
                result.negative_similarities.append(similarity)


# ---------------------------------------------------------------------- reports
def _leg_row(label: str, leg: LegScore) -> str:
    return (
        f"{label:<28} {leg.recall:>9.2f} {leg.mrr:>6.3f} {leg.ndcg:>7.3f} "
        f"{leg.p_at_5:>5.2f} {leg.fp_rate:>5.2f}"
    )


def format_results(results: Sequence[ModelResult]) -> str:
    """The comparison table used to choose the final models: the full pipeline per model, with
    the vector leg alone beneath it."""
    header = (
        f"{'model':<28} {'recall@10':>9} {'MRR':>6} {'nDCG@10':>7} {'P@5':>5} {'FP':>5} "
        f"{'files':>6} {'index s':>8} {'p50 ms':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: (r.error != "", -r.full.ndcg)):
        if r.error:
            lines.append(f"{r.model:<28} FAILED: {r.error}")
            continue
        lines.append(
            f"{_leg_row(r.model, r.full)} {r.files:>6} {r.index_seconds:>8.1f} {r.p50_ms:>7.0f}"
        )
        lines.append(_leg_row("  vector leg only", r.vector))
        if r.full.missed:
            lines.append("    missed: " + "; ".join(r.full.missed[:5]))
        if r.full.false_positives:
            lines.append("    false positives: " + "; ".join(r.full.false_positives[:5]))
    return "\n".join(lines)


def _percentile_row(label: str, values: Sequence[float]) -> str:
    cells = " ".join(f"{v:>6.3f}" for v in percentiles(values)) or "   (none)"
    return f"  {label:<12} {len(values):>4}  {cells}"


def format_calibration(results: Sequence[ModelResult]) -> str:
    """Cosine-similarity percentiles of true hits vs. negatives per model.

    A good similarity floor sits above most negatives and below most true hits; when the two
    ranges overlap heavily the model cannot tell them apart on similarity alone.
    """
    header = f"  {'':<12} {'n':>4}  " + " ".join(f"{'p' + str(p):>6}" for p in PERCENTILES)
    lines: list[str] = []
    for r in results:
        if r.error:
            continue
        lines += [r.model, header]
        lines.append(_percentile_row("true hits", r.true_similarities))
        lines.append(_percentile_row("negatives", r.negative_similarities))
    return "\n".join(lines)
