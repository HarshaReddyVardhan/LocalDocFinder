"""Document-type classification without an LLM.

Evidence comes from file names, section headings, keywords, the path, and (optionally) the
similarity of the document vector to a per-type prototype vector. Rules live in settings, so
a new rule is a config change; a new *type* with custom logic is one file with a decorator.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from vector_embed.core.registry import Registry, discover_modules
from vector_embed.core.settings import DocTypeRule, DocTypeSettings, ScopeSettings

DOC_TYPE_OTHER = "other"
DOC_TYPE_CODE = "code"

_FILENAME_WEIGHT = 0.5
_HEADING_WEIGHT = 0.15
_HEADING_CAP = 0.5
_KEYWORD_WEIGHT = 0.1
_KEYWORD_CAP = 0.3
_PATH_WEIGHT = 0.4
_PROTO_FLOOR = 0.35  # cosine below this carries no signal
_PROTO_SPAN = 0.45  # cosine floor + span (0.8) counts as full evidence
_HEAD_LINES = 400  # headings and keywords are looked for in the start of the document


@dataclass(frozen=True)
class DocInfo:
    path: Path
    title: str
    text: str
    vector: np.ndarray | None = None  # document-level embedding, when available


@dataclass(frozen=True)
class Classification:
    doc_type: str
    confidence: float
    scores: Mapping[str, float] = field(default_factory=dict)


class DocTypeClassifier(ABC):
    """One class per document type."""

    name: ClassVar[str]
    priority: ClassVar[int] = 100  # tie-break: lower wins

    def __init__(self, settings: DocTypeSettings) -> None:
        self.settings = settings

    @abstractmethod
    def score(self, info: DocInfo) -> float:
        """Rule-based evidence in 0..1 that ``info`` is this type."""
        ...


class RuleBasedClassifier(DocTypeClassifier):
    """Scores a document with the regex rules configured for ``name``."""

    def _rule(self) -> DocTypeRule:
        return self.settings.rules.get(self.name, DocTypeRule())

    def score(self, info: DocInfo) -> float:
        rule = self._rule()
        head = "\n".join(info.text.splitlines()[:_HEAD_LINES])
        lines = [line.strip().strip("#*:").strip() for line in head.splitlines()]
        total = 0.0
        if _any_match(rule.filename, info.path.stem):
            total += _FILENAME_WEIGHT
        if _any_match(rule.path, info.path.as_posix()):
            total += _PATH_WEIGHT
        heading_hits = sum(
            1 for pat in rule.headings if any(re.fullmatch(pat, ln, re.I) for ln in lines if ln)
        )
        total += min(_HEADING_CAP, heading_hits * _HEADING_WEIGHT)
        keyword_hits = sum(1 for pat in rule.keywords if re.search(pat, head, re.I))
        total += min(_KEYWORD_CAP, keyword_hits * _KEYWORD_WEIGHT)
        return min(1.0, total)


def _any_match(patterns: tuple[str, ...], value: str) -> bool:
    return any(re.search(p, value, re.I) for p in patterns)


DOCTYPES: Registry[type[DocTypeClassifier]] = Registry("doctype")
register_doctype = DOCTYPES.register

_BUILTIN_PACKAGE = "vector_embed.core.doctypes"


class DocTypeClassifierSet:
    """All registered classifiers for one settings object."""

    def __init__(
        self,
        settings: DocTypeSettings,
        scope: ScopeSettings,
        prototypes: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        discover_modules(_BUILTIN_PACKAGE)
        self.settings = settings
        self._code_exts = scope.code_exts
        self._prototypes = dict(prototypes or {})
        self._classifiers = sorted(
            (cls(settings) for cls in DOCTYPES), key=lambda c: (c.priority, c.name)
        )

    def classify(self, info: DocInfo) -> Classification:
        """Best type for ``info``; ``other`` when no type reaches the threshold."""
        if info.path.suffix.lower() in self._code_exts:
            return Classification(DOC_TYPE_CODE, 1.0, {DOC_TYPE_CODE: 1.0})
        scores: dict[str, float] = {}
        for classifier in self._classifiers:
            total = classifier.score(info) + self._prototype_bonus(classifier.name, info)
            scores[classifier.name] = min(1.0, total)
        best = max(self._classifiers, key=lambda c: (scores[c.name], -c.priority), default=None)
        if best is None or scores[best.name] < self.settings.threshold:
            return Classification(DOC_TYPE_OTHER, 1.0 - max(scores.values(), default=0.0), scores)
        return Classification(best.name, scores[best.name], scores)

    def _prototype_bonus(self, name: str, info: DocInfo) -> float:
        proto = self._prototypes.get(name)
        if proto is None or info.vector is None:
            return 0.0
        norm = float(np.linalg.norm(info.vector) * np.linalg.norm(proto))
        if norm == 0:
            return 0.0
        cosine = float(np.dot(info.vector, proto)) / norm
        strength = min(1.0, max(0.0, (cosine - _PROTO_FLOOR) / _PROTO_SPAN))
        return self.settings.prototype_weight * strength


PROTOTYPE_TEXTS: dict[str, str] = {
    "resume": "Resume. Work experience, education, skills, projects, certifications, "
    "software engineer, responsibilities, achievements, technologies used.",
    "cover_letter": "Dear hiring manager, I am writing to apply for the position. "
    "I am excited about this opportunity. Sincerely.",
    "jd": "Job description. We are hiring. Responsibilities, requirements, qualifications, "
    "years of experience, nice to have, benefits, apply now.",
    "invoice": "Invoice number, bill to, date, due date, description, quantity, unit price, "
    "subtotal, tax, total amount due, payment terms.",
    "paper": "Abstract. Introduction. Related work. Method. Experiments. Results. "
    "Conclusion. References. We propose a novel approach.",
    "plan": "Plan. Context. Steps. Build order. Verification. Implementation plan "
    "with tasks and files to change.",
    "notes": "Notes and memory. Remember this. Meeting notes, decisions, todo list.",
}


def build_prototypes(
    embed: Callable[[list[str]], np.ndarray], texts: Mapping[str, str] | None = None
) -> dict[str, np.ndarray]:
    """Embed the prototype texts once; pass the result to ``DocTypeClassifierSet``."""
    source = dict(texts or PROTOTYPE_TEXTS)
    names = list(source)
    vectors = embed([source[n] for n in names])
    return {name: vectors[i] for i, name in enumerate(names)}
