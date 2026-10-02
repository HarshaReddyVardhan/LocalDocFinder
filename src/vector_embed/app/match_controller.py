"""Qt-free logic behind the Match panel: the run state, live token footer, and step helpers."""

from collections.abc import Callable, Iterator
from pathlib import Path

from vector_embed.app.assistant import ChatState
from vector_embed.core.match.judge import MatchError
from vector_embed.core.match.pipeline import DocumentScore, MatchPipeline, MatchRun
from vector_embed.core.match.recall import MatchCandidate, select_top, selected
from vector_embed.core.match.scoring import Requirement
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.match import pipeline_of
from vector_embed.core.tokens import estimate_tokens

LOCAL = "🖥 local"
_TOKENS_PER_REQUIREMENT = 15


def format_tokens(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


class MatchController:
    """One match at a time; the window drives the steps and shows ``footer`` as a live total."""

    def __init__(
        self,
        context_factory: Callable[[], SkillContext],
        destination: Callable[[MatchCandidate], str] = lambda _candidate: LOCAL,
    ) -> None:
        self._factory = context_factory
        self._destination = destination
        self._pipeline: MatchPipeline | None = None
        self.run: MatchRun | None = None

    @property
    def pipeline(self) -> MatchPipeline:
        if self._pipeline is None:
            self._pipeline = pipeline_of(self._factory())
        return self._pipeline

    # ------------------------------------------------------------------ steps
    def start(
        self, jd_text: str, doc_type: str | None = None, all_versions: bool = False
    ) -> MatchRun:
        self.run = self.pipeline.start(jd_text, doc_type, all_versions)
        return self.run

    def checklist(self) -> list[Requirement]:
        return self.pipeline.build_checklist(self._require())

    def set_checklist(self, requirements: list[Requirement]) -> None:
        """Store the user's edits (ticked, un-ticked, re-weighted requirements)."""
        self._require().requirements = requirements

    def score(self, progress: Callable[[str], None] | None = None) -> list[DocumentScore]:
        return self.pipeline.score(self._require(), progress)

    def verdict(self) -> Iterator[str]:
        return self.pipeline.stream_verdict(self._require())

    def add_file(self, path: str) -> MatchCandidate:
        run = self._require()
        candidate = self.pipeline.recall_step.add_file(path, run.jd_text)
        if all(c.path != candidate.path for c in run.candidates):
            run.candidates.append(candidate)
        return candidate

    def top(self, n: int) -> None:
        select_top(self._require().candidates, n)

    def chat_state(self, top: int | None = None) -> ChatState:
        """Pinned resumes and scratch text (JD + step-3 results) for the follow-up chat."""
        pinned, scratch = self.pipeline.chat_context(self._require(), top)
        return ChatState(pinned=pinned, scratch=scratch)

    # ------------------------------------------------------------------ display helpers
    def footer(self) -> str:
        """``JD + 3 documents ≈ 7.1k tokens → local`` for what Score would send."""
        run = self._require()
        chosen = selected(run.candidates)
        if not chosen:
            return "nothing selected"
        jd = estimate_tokens(run.jd_text)
        checklist = len(run.requirements) * _TOKENS_PER_REQUIREMENT
        total = jd + sum(c.tokens + checklist + jd for c in chosen)
        where = sorted({self._destination(c) for c in chosen})
        locked = sum(c.locked for c in chosen)
        note = f" · 🔒 {locked} scored locally" if locked else ""
        noun = "document" if len(chosen) == 1 else "documents"
        return (
            f"JD + {len(chosen)} {noun} ≈ {format_tokens(total)} tokens → {' / '.join(where)}{note}"
        )

    def candidate_row(self, candidate: MatchCandidate) -> list[str]:
        """Cell texts for the candidate table (after the checkbox column)."""
        from datetime import datetime

        badge = f"{candidate.versions} versions" if candidate.versions > 1 else ""
        if candidate.versions > 1 and candidate.is_latest:
            badge += ", newest"
        elif not candidate.is_latest:
            badge = "older version"
        folder = str(Path(candidate.path).parent)
        stamp = datetime.fromtimestamp(candidate.modified_at).strftime("%Y-%m-%d")
        where = ("🔒 " if candidate.locked else "") + self._destination(candidate)
        return [
            candidate.name,
            folder,
            stamp,
            badge,
            f"{candidate.similarity:.2f}",
            format_tokens(candidate.tokens),
            where,
        ]

    def _require(self) -> MatchRun:
        if self.run is None:
            raise MatchError("paste a job description first")
        return self.run
