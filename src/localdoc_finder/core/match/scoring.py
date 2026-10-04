"""Deterministic scoring for Match.

The LLM only judges each requirement (met / partial / missing + a quote). The numeric score is
computed here, so it is reproducible and explainable ("7/9 must-haves met"). Quotes are checked
against the document: a claimed match without real evidence is downgraded to ``unverified``.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from localdoc_finder.core.settings import MatchSettings

Status = Literal["met", "partial", "missing", "unverified"]
Kind = Literal["must", "nice"]

STATUS_CREDIT: dict[str, float] = {"met": 1.0, "partial": 0.5, "missing": 0.0, "unverified": 0.0}
_MIN_QUOTE_CHARS = 8  # below this a quote is checked as exact whole words, not fuzzily
_MIN_SHORT_QUOTE_CHARS = 2  # a single letter proves nothing
_MIN_TERM_CHARS = 2
_MIN_SHARED_WORDS = 0.5  # a near-verbatim quote shares at least half its words with its source
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "the",
        "as",
        "is",
        "are",
        "be",
        "years",
        "year",
        "experience",
        "strong",
        "good",
        "knowledge",
        "working",
        "skills",
        "skill",
        "ability",
        "plus",
        "least",
    ]
)
_NON_WORD = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


class Requirement(BaseModel):
    """One line of the checklist extracted from the job description."""

    id: int
    text: str
    kind: Kind = "must"
    category: Literal["skill", "experience", "education", "cert", "other"] = "skill"
    years: int | None = None
    enabled: bool = True  # the user can untick a requirement before scoring
    weight: float = Field(default=1.0, gt=0)  # and re-weight it


@dataclass(frozen=True)
class RowResult:
    requirement_id: int
    status: Status
    evidence: str = ""

    @property
    def verified(self) -> bool:
        return self.status != "unverified"


@dataclass
class ScoreBreakdown:
    score: int  # 0-100
    met_must: int
    total_must: int
    met_nice: int
    total_nice: int
    unverified: int = 0
    rows: list[RowResult] = field(default_factory=list)

    @property
    def summary_line(self) -> str:
        line = f"{self.met_must}/{self.total_must} must-haves met"
        if self.total_nice:
            line += f", {self.met_nice}/{self.total_nice} nice-to-haves"
        if self.unverified:
            line += f", {self.unverified} unverified"
        return line


def normalise(text: str) -> str:
    """Lower-case, punctuation-free, single-spaced: the form quotes are compared in."""
    return _SPACES.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def key_terms(requirement: str) -> frozenset[str]:
    """The words of a requirement that a quote must not swap out ("Kubernetes", "SQL", "5")."""
    return frozenset(
        word
        for word in normalise(requirement).split()
        if word not in _STOPWORDS and (len(word) >= _MIN_TERM_CHARS or word.isdigit())
    )


def _short_quote_in_text(quote: str, text: str) -> bool:
    """A short quote ("Go", "AWS", "C++") counts only as whole words, exactly as written.

    Fuzzy matching is meaningless at this length, and the punctuation is often the point.
    """
    wanted = _SPACES.sub(" ", quote.lower()).strip()
    if not any(char.isalnum() for char in wanted) or len(wanted) < _MIN_SHORT_QUOTE_CHARS:
        return False
    pattern = r"(?<!\w)" + re.escape(wanted).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


def quote_in_text(
    quote: str, text: str, threshold: float = 0.85, terms: frozenset[str] = frozenset()
) -> bool:
    """Whether ``quote`` really appears in ``text`` (whitespace/case/punctuation-insensitive).

    A near match must still contain every word of ``terms`` the quote uses, so a quote that
    swaps one technology for another ("MySQL" for "PostgreSQL") is not accepted as close enough.
    """
    wanted = normalise(quote)
    if len(wanted) < _MIN_QUOTE_CHARS:
        return _short_quote_in_text(quote, text)
    haystack = normalise(text)
    if wanted in haystack:
        return True
    quote_words = set(wanted.split())
    needed = terms.intersection(quote_words)
    min_shared = max(1, int(len(quote_words) * _MIN_SHARED_WORDS))
    words = haystack.split()
    size = len(wanted.split())
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(wanted)  # the quote's index is built once, not per window
    for start in range(max(1, len(words) - size + 1)):
        window_words = words[start : start + size + 1]
        if not needed.issubset(window_words) or (
            len(quote_words.intersection(window_words)) < min_shared
        ):
            continue
        matcher.set_seq1(" ".join(window_words))
        # the quick ratios are cheap upper bounds that rule out most windows
        if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
            continue
        if matcher.ratio() >= threshold:
            return True
    return False


def verify_rows(
    rows: list[RowResult],
    document_text: str,
    settings: MatchSettings,
    requirements: list[Requirement] | None = None,
) -> list[RowResult]:
    """Downgrade met/partial rows whose evidence quote is not found in the document."""
    terms = {req.id: key_terms(req.text) for req in requirements or []}
    verified: list[RowResult] = []
    for row in rows:
        claims_match = row.status in ("met", "partial")
        if claims_match and not quote_in_text(
            row.evidence,
            document_text,
            settings.evidence_threshold,
            terms.get(row.requirement_id, frozenset()),
        ):
            verified.append(RowResult(row.requirement_id, "unverified", row.evidence))
        else:
            verified.append(row)
    return verified


def compute_score(
    requirements: list[Requirement], rows: list[RowResult], settings: MatchSettings
) -> ScoreBreakdown:
    """Weighted share of satisfied requirements, 0-100. Only enabled requirements count."""
    by_id = {row.requirement_id: row for row in rows}
    earned = possible = 0.0
    met = {"must": 0, "nice": 0}
    totals = {"must": 0, "nice": 0}
    unverified = 0
    used: list[RowResult] = []
    for req in requirements:
        if not req.enabled:
            continue
        row = by_id.get(req.id, RowResult(req.id, "missing"))
        used.append(row)
        base = settings.must_weight if req.kind == "must" else settings.nice_weight
        weight = base * req.weight
        possible += weight
        earned += weight * STATUS_CREDIT[row.status]
        totals[req.kind] += 1
        if row.status == "met":
            met[req.kind] += 1
        if row.status == "unverified":
            unverified += 1
    score = round(100 * earned / possible) if possible else 0
    return ScoreBreakdown(
        score=score,
        met_must=met["must"],
        total_must=totals["must"],
        met_nice=met["nice"],
        total_nice=totals["nice"],
        unverified=unverified,
        rows=used,
    )
