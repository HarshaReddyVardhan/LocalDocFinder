"""Qt-free logic behind the Match panel: the run state, live token footer, and step helpers."""

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from localdoc_finder.app.assistant import ChatState, CloudPreview
from localdoc_finder.core.match.judge import MatchError
from localdoc_finder.core.match.pipeline import DocumentScore, MatchPipeline, MatchRun
from localdoc_finder.core.match.recall import MatchCandidate, select_top, selected
from localdoc_finder.core.match.scoring import Requirement
from localdoc_finder.core.models.catalog import ROLE_MATCH_SCORER
from localdoc_finder.core.runtime import CloudContext
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.skills.match import pipeline_of
from localdoc_finder.core.tokens import estimate_tokens

LOCAL = "🖥 local"
# The cloud provider adds one line, after masking, when a model rejects JSON-schema mode.
JSON_FALLBACK_NOTE = (
    "\n\n--- note ---\nIf the model does not support JSON schemas, one extra line is added to "
    "each request: 'Reply with JSON matching: <the response schema>'. It contains no "
    "document text."
)
_ROUTE_TTL = 5.0  # seconds a worked-out route is reused
_TOKENS_PER_REQUIREMENT = 15
_OUTPUT_TOKENS_PER_DOCUMENT = 400


def _key(path: str) -> str:
    return os.path.normcase(path)  # Windows paths ignore case


def format_tokens(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


class MatchController:
    """One match at a time; the window drives the steps and shows ``footer`` as a live total."""

    def __init__(
        self,
        context_factory: Callable[[], SkillContext],
        destination: Callable[[MatchCandidate], str] | None = None,
    ) -> None:
        self._factory = context_factory
        self._destination = destination or self._default_destination
        self._pipeline: MatchPipeline | None = None
        self._destination_cache: tuple[float, str] | None = None
        self._choices: dict[str, bool] = {}  # ticks the user set, by path, kept between runs
        self._redact: bool | None = None  # "Remove personal details" chosen in the panel
        self.run: MatchRun | None = None

    def reset_context(self) -> None:
        """A setting changed: rebuild the pipeline on next use (a finished run stays visible)."""
        self._pipeline = None
        self._destination_cache = None

    @property
    def pipeline(self) -> MatchPipeline:
        if self._pipeline is None:
            self._pipeline = pipeline_of(self._factory())
        return self._pipeline

    # ------------------------------------------------------------------ cloud consent
    def _cloud(self) -> CloudContext | None:
        cloud = self._factory().extras.get("cloud")
        return cloud if isinstance(cloud, CloudContext) else None

    def cloud_preview(self, step: str = "score") -> CloudPreview | None:
        """Exactly what ``step`` ("checklist" or "score") would send to the cloud.

        ``None`` when the step stays on this machine.
        """
        run = self._require()
        messages = (
            self.pipeline.checklist_cloud_messages(run)
            if step == "checklist"
            else self.pipeline.cloud_messages(run)
        )
        cloud = self._cloud()
        if not messages or cloud is None or cloud.provider is None:
            return None
        destination = str(cloud.router.destination(ROLE_MATCH_SCORER))
        outbound = cloud.provider.prepare(messages)
        privacy = cloud.privacy
        remote = (
            1 if step == "checklist" else sum(1 for c in selected(run.candidates) if not c.locked)
        )
        return CloudPreview(
            destination,
            privacy.badge(outbound, destination, remote),
            privacy.shield_note(outbound),
            privacy.preview(outbound) + JSON_FALLBACK_NOTE,
        )

    def grant_cloud_consent(self) -> None:
        """The user pressed Send in the preview: allow cloud calls for this run."""
        cloud = self._cloud()
        if cloud is not None:
            cloud.consent.grant()

    def revoke_cloud_consent(self) -> None:
        cloud = self._cloud()
        if cloud is not None:
            cloud.consent.revoke()

    # ------------------------------------------------------------------ steps
    def start(
        self, jd_text: str, doc_type: str | None = None, all_versions: bool = False
    ) -> MatchRun:
        self.revoke_cloud_consent()  # consent belongs to one run, never the next
        self._destination_cache = None
        self._apply_redaction()
        run = self.pipeline.start(jd_text, doc_type, all_versions)
        for candidate in run.candidates:  # the user's earlier ticks win over the default
            choice = self._choices.get(_key(candidate.path))
            if choice is not None:
                candidate.selected = choice
        self.run = run
        return run

    # ------------------------------------------------------------------ selection
    def choose(self, candidate: MatchCandidate, selected_: bool) -> None:
        """Tick or untick a document; remembered for later runs in this session."""
        candidate.selected = selected_
        self._choices[_key(candidate.path)] = selected_

    def choose_all(self, selected_: bool) -> None:
        for candidate in self._require().candidates:
            self.choose(candidate, selected_)

    # ------------------------------------------------------------------ personal details
    @property
    def redacts_personal(self) -> bool:
        """The checkbox state: the panel's choice, else the saved setting (once known)."""
        if self._redact is not None:
            return self._redact
        if self._pipeline is None:
            return False  # not loaded yet; the panel syncs after the first recall
        cloud = self._cloud()
        return cloud.privacy.redacts_personal if cloud is not None else False

    def set_redact_personal(self, enabled: bool) -> None:
        """Remove names, emails, phones and addresses from what Match sends to the cloud."""
        self._redact = enabled
        if self._pipeline is not None:  # else applied when the next run starts
            self._apply_redaction()

    def _apply_redaction(self) -> None:
        cloud = self._cloud()
        if self._redact is not None and cloud is not None:
            cloud.privacy.set_redact_personal(self._redact)

    def _ensure_session(self) -> None:
        """One chat session for the whole match, so the model loads once instead of per call.

        The window ends it (and unloads the model) when the user leaves Match or closes it.
        """
        gateway = self.pipeline.gateway
        if not gateway.session_active:
            gateway.begin_chat()

    def finish(self) -> None:
        """Unload the model now that the match is over."""
        gateway = self.pipeline.gateway
        if gateway.session_active:
            gateway.end_chat("match finished")

    def checklist(self) -> list[Requirement]:
        self._ensure_session()
        return self.pipeline.build_checklist(self._require(), session=True)

    def set_checklist(self, requirements: list[Requirement]) -> None:
        """Store the user's edits (ticked, un-ticked, re-weighted requirements)."""
        self._require().requirements = requirements

    def score(self, progress: Callable[[str], None] | None = None) -> list[DocumentScore]:
        self._ensure_session()
        return self.pipeline.score(self._require(), progress, session=True)

    def verdict(self) -> Iterator[str]:
        self._ensure_session()
        return self.pipeline.stream_verdict(self._require(), session=True)

    def add_file(self, path: str) -> MatchCandidate:
        run = self._require()
        candidate = self.pipeline.recall_step.add_file(path, run.jd_text)
        if all(c.path != candidate.path for c in run.candidates):
            run.candidates.append(candidate)
        return candidate

    def top(self, n: int) -> None:
        candidates = self._require().candidates
        select_top(candidates, n)
        for candidate in candidates:
            self._choices[_key(candidate.path)] = candidate.selected

    def chat_state(self, top: int | None = None) -> ChatState:
        """Pinned resumes and scratch text (JD + step-3 results) for the follow-up chat.

        By default every document that was ticked and scored is pinned; ``top`` limits it to the
        best few."""
        pinned, scratch = self.pipeline.chat_context(self._require(), top)
        return ChatState(pinned=pinned, scratch=scratch)

    # ------------------------------------------------------------------ display helpers
    def _default_destination(self, candidate: MatchCandidate) -> str:
        """Where this document would be judged: locked files always stay local."""
        if candidate.locked:
            return LOCAL
        return self._open_destination()

    def _open_destination(self) -> str:
        """The route for an ordinary document, worked out once and reused for every table row.

        Resolving it checks the model registry and the hardware; doing that per row, on each
        redraw and footer update, made the table sluggish.
        """
        now = time.monotonic()
        if self._destination_cache is not None and now - self._destination_cache[0] < _ROUTE_TTL:
            return self._destination_cache[1]
        cloud = self.pipeline.gateway.cloud_destination(ROLE_MATCH_SCORER)
        label = f"☁ {cloud}" if cloud else LOCAL
        self._destination_cache = (now, label)
        return label

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
        remote = [c for c in chosen if not c.locked]
        cost = self.pipeline.gateway.estimate_cost(
            ROLE_MATCH_SCORER,
            sum(c.tokens + checklist + jd for c in remote) + jd,
            _OUTPUT_TOKENS_PER_DOCUMENT * len(remote),
        )
        price = f" · est. ${cost:.2f}" if cost > 0 else ""
        noun = "document" if len(chosen) == 1 else "documents"
        return (
            f"JD + {len(chosen)} {noun} ≈ {format_tokens(total)} tokens "
            f"→ {' / '.join(where)}{price}{note}"
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
