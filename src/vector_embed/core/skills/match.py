"""Match skill: rank a group of documents against pasted text (the job-description example).

``ve match --jd-file jd.txt`` runs every step with sensible defaults; the desktop UI drives the
same ``MatchPipeline`` step by step so the user can edit the checklist and the candidate list.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field

from vector_embed.core.documents import DocumentLoader
from vector_embed.core.match.pipeline import DocumentScore, MatchPipeline, MatchRun
from vector_embed.core.match.recall import select_top
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_MATCH_SCORER
from vector_embed.core.skills.ask import gateway_of, privacy_of
from vector_embed.core.skills.base import (
    UI_TABLE,
    Skill,
    SkillContext,
    SkillInput,
    register_skill,
)

_CUT = " (reduced: cut to fit)"


class MatchInput(SkillInput):
    jd_file: str | None = Field(default=None, description="File holding the text to match")
    jd: str | None = Field(default=None, description="The text to match, pasted inline")
    doc_type: str = Field(default="resume", description="Kind of documents to rank")
    top: int | None = Field(default=None, ge=1, description="Score only the N most similar")
    all_versions: bool = Field(default=False, description="Include older versions")


def pipeline_of(ctx: SkillContext) -> MatchPipeline:
    loader = ctx.extras.get("documents")
    if not isinstance(loader, DocumentLoader):
        raise RuntimeError("no document loader configured for this context")
    return MatchPipeline(ctx, gateway_of(ctx), loader)


def format_scores(run: MatchRun) -> str:
    """Ranked table: score, must-haves, file, notes."""
    lines = []
    for rank, item in enumerate(run.ranked(), 1):
        if item.breakdown is None:
            lines.append(f"{rank:>2}. ----  {item.candidate.name}: {item.error}")
            continue
        flag = " ⚠" if item.breakdown.unverified else ""
        cut = _CUT if item.judgement and item.judgement.reduced else ""
        lines.append(
            f"{rank:>2}. {item.score:>3}  {item.candidate.name}  "
            f"[{item.breakdown.summary_line}]{flag}{cut}"
        )
    return "\n".join(lines)


@register_skill("match")
class MatchSkill(Skill):
    name = "match"
    title = "Match"
    description = "Rank your documents (e.g. resumes) against a pasted job description."
    Input = MatchInput
    roles = (ROLE_MATCH_SCORER, ROLE_CHAT)
    ui_hint = UI_TABLE

    def __init__(self, ctx: SkillContext) -> None:
        super().__init__(ctx)
        self.pipeline = pipeline_of(ctx)

    def _read(self, params: MatchInput) -> str:
        if params.jd_file:
            document = self.pipeline._loader.load(Path(params.jd_file))
            privacy = privacy_of(self.ctx)
            if (
                privacy is not None
                and privacy.is_never_send(document.path, document.doc_type or None)
                and self.pipeline.gateway.will_use_cloud(ROLE_MATCH_SCORER)
            ):
                raise RuntimeError(
                    f"{Path(document.path).name} is private and cannot be sent to a cloud model"
                )
            return document.text
        if params.jd and params.jd.strip():
            return params.jd
        raise RuntimeError("provide the text to match with --jd-file or --jd")

    def prepare(self, params: MatchInput) -> MatchRun:
        """Steps 1-2 with defaults: recall, tick by threshold (or top N)."""
        run = self.pipeline.start(self._read(params), params.doc_type, params.all_versions)
        if params.top:
            select_top(run.candidates, params.top)
        return run

    @contextmanager
    def _session(self) -> Iterator[None]:
        """Load the model once for all the calls of a match, and unload it when finished."""
        gateway = self.pipeline.gateway
        began = not gateway.session_active
        if began:
            gateway.begin_chat()
        try:
            yield
        finally:
            if began:
                gateway.end_chat("match finished")

    def run(self, params: SkillInput) -> MatchRun:
        assert isinstance(params, MatchInput)
        run = self.prepare(params)
        if any(c.selected for c in run.candidates):
            with self._session():
                self.pipeline.score(run, session=True)
        return run

    def stream(self, params: SkillInput) -> Iterator[str]:
        assert isinstance(params, MatchInput)
        run = self.prepare(params)
        yield f"Recalled {len(run.candidates)} candidate(s); "
        yield f"{sum(c.selected for c in run.candidates)} ticked.\n"
        if not any(c.selected for c in run.candidates):
            yield "Nothing to score: no candidate passed the similarity threshold.\n"
            return
        messages: list[str] = []
        self.pipeline.score(run, progress=messages.append)
        yield "\n".join(messages) + "\n\n"
        yield format_scores(run) + "\n\n"
        yield from self.pipeline.stream_verdict(run)
        yield "\n"

    def render(self, output: object) -> str:
        assert isinstance(output, MatchRun)
        return format_scores(output) if output.scores else "no documents scored"


__all__ = ["DocumentScore", "MatchInput", "MatchSkill", "format_scores", "pipeline_of"]
