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

from vector_embed.core.settings import MatchSettings

Status = Literal["met", "partial", "missing", "unverified"]
Kind = Literal["must", "nice"]

STATUS_CREDIT: dict[str, float] = {"met": 1.0, "partial": 0.5, "missing": 0.0, "unverified": 0.0}
_MIN_QUOTE_CHARS = 8
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


def quote_in_text(quote: str, text: str, threshold: float = 0.85) -> bool:
    """Whether ``quote`` really appears in ``text`` (whitespace/case/punctuation-insensitive)."""
    wanted = normalise(quote)
    if len(wanted) < _MIN_QUOTE_CHARS:
        return False
    haystack = normalise(text)
    if wanted in haystack:
        return True
    words = haystack.split()
    size = len(wanted.split())
    for start in range(max(1, len(words) - size + 1)):
        window = " ".join(words[start : start + size + 1])
        matcher = difflib.SequenceMatcher(None, window, wanted, autojunk=False)
        if matcher.ratio() >= threshold:
            return True
    return False


def verify_rows(
    rows: list[RowResult], document_text: str, settings: MatchSettings
) -> list[RowResult]:
    """Downgrade met/partial rows whose evidence quote is not found in the document."""
    verified: list[RowResult] = []
    for row in rows:
        claims_match = row.status in ("met", "partial")
        if claims_match and not quote_in_text(
            row.evidence, document_text, settings.evidence_threshold
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
