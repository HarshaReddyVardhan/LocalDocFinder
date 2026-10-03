"""The Match workflow: recall -> checklist (once) -> score each ticked document -> verdict -> chat.

What leaves the machine (when a cloud provider is used, see ``core/privacy``): step 1 never
(local embeddings only); step 2 the job description only; step 3 the checklist plus ONE
document per call; step 4 only the step-3 JSON. Locked documents are always judged locally.
"""

import json
import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from vector_embed.core.documents import DocumentError, DocumentLoader
from vector_embed.core.llm import ChatBlockedError, LlmGateway
from vector_embed.core.match import judge
from vector_embed.core.match.judge import Judgement, MatchError
from vector_embed.core.match.recall import MatchCandidate, Recall, selected
from vector_embed.core.match.scoring import Requirement, ScoreBreakdown, compute_score
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_MATCH_SCORER
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.prompt_safety import fence_for
from vector_embed.core.providers.base import Message, ProviderError, ProviderUnavailableError
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.tokens import fit_to_budget

logger = logging.getLogger(__name__)

VERDICT_SYSTEM = (
    "You compare candidate documents for a role using ONLY the scored data provided. Rank them "
    "from best to worst and explain each in one or two sentences, naming what is missing. Do "
    "not invent facts that are not in the data. Be concise."
)
_JD_SUMMARY_TOKENS = 300
_CLOUD_WORKERS = 4


@dataclass
class DocumentScore:
    candidate: MatchCandidate
    breakdown: ScoreBreakdown | None = None
    judgement: Judgement | None = None
    error: str = ""

    @property
    def score(self) -> int:
        return self.breakdown.score if self.breakdown else -1


@dataclass
class MatchRun:
    """All state of one match: the text, the candidate list the user edits, and the results."""

    jd_text: str
    doc_type: str
    candidates: list[MatchCandidate]
    requirements: list[Requirement] = field(default_factory=list)
    scores: list[DocumentScore] = field(default_factory=list)

    def ranked(self) -> list[DocumentScore]:
        return sorted(
            self.scores,
            key=lambda s: (
                s.score,
                s.breakdown.met_must if s.breakdown else 0,
                s.candidate.similarity,
            ),
            reverse=True,
        )


Progress = Callable[[str], None]


class MatchPipeline:
    def __init__(self, ctx: SkillContext, gateway: LlmGateway, loader: DocumentLoader) -> None:
        self._ctx = ctx
        self._gateway = gateway
        self._loader = loader
        privacy = ctx.extras.get("privacy")
        is_locked: Callable[[str], bool] = (
            privacy.is_never_send if isinstance(privacy, PrivacyFilter) else (lambda _path: False)
        )
        self.recall_step = Recall(ctx, loader, is_locked)

    @property
    def gateway(self) -> LlmGateway:
        return self._gateway

    # ------------------------------------------------------------------ step 1
    def start(
        self, jd_text: str, doc_type: str | None = None, include_old_versions: bool = False
    ) -> MatchRun:
        """Recall candidates (local only). Nothing is sent to any chat model yet."""
        kind = self._ctx.settings.match.default_doc_type if doc_type is None else doc_type
        return MatchRun(jd_text, kind, self.recall_step.recall(jd_text, kind, include_old_versions))

    # ------------------------------------------------------------------ step 2
    def build_checklist(self, run: MatchRun, session: bool = False) -> list[Requirement]:
        """Extract the requirement checklist once; later calls reuse (and keep user edits)."""
        if not run.requirements:
            settings = self._ctx.settings
            run.requirements = judge.extract_requirements(
                self._gateway, run.jd_text, settings.match, session, chat=settings.chat
            )
        return run.requirements

    # ------------------------------------------------------------------ step 3
    def checklist_cloud_messages(self, run: MatchRun) -> list[Message]:
        """The checklist request, if building it would go to the cloud (else nothing)."""
        if run.requirements or not self._gateway.will_use_cloud(ROLE_MATCH_SCORER):
            return []
        settings = self._ctx.settings
        return judge.requirements_messages(run.jd_text, settings.match, settings.chat)

    def cloud_messages(self, run: MatchRun) -> list[Message]:
        """Every judge prompt that scoring would send to the cloud (none if it stays local).

        Locked (private) documents are scored locally and so never appear here. Needs the
        checklist to exist already: building it is its own (previewed) step.
        """
        requirements = run.requirements
        if not requirements or not self._gateway.will_use_cloud(ROLE_MATCH_SCORER):
            return []
        settings = self._ctx.settings
        messages: list[Message] = []
        for candidate in selected(run.candidates):
            if candidate.locked:
                continue
            document = self._loader.load(candidate.path)
            prompt, _ = judge.prepare_judge_messages(
                requirements,
                run.jd_text,
                name=candidate.name,
                document_text=document.text,
                match=settings.match,
                chat=settings.chat,
            )
            messages.extend(prompt)
        return messages

    def score(
        self, run: MatchRun, progress: Progress | None = None, session: bool = False
    ) -> list[DocumentScore]:
        """Judge every ticked candidate against the checklist, one call per document."""
        chosen = selected(run.candidates)
        if not chosen:
            raise MatchError("no documents are selected")
        requirements = self.build_checklist(run, session)
        if self._gateway.will_use_cloud(ROLE_MATCH_SCORER):
            run.scores = self._score_in_parallel(run, requirements, chosen, progress, session)
        else:
            run.scores = []
            for position, candidate in enumerate(chosen, 1):
                if progress:
                    progress(f"scoring {position}/{len(chosen)}: {candidate.name}")
                run.scores.append(self._score_one(run, requirements, candidate, session))
        return run.scores

    def _score_in_parallel(
        self,
        run: MatchRun,
        requirements: list[Requirement],
        chosen: list[MatchCandidate],
        progress: Progress | None,
        session: bool,
    ) -> list[DocumentScore]:
        """Cloud judging runs concurrently; locked documents are then judged locally, in turn."""
        results: dict[str, DocumentScore] = {}
        remote = [c for c in chosen if not c.locked]
        done = 0
        with ThreadPoolExecutor(max_workers=_CLOUD_WORKERS) as pool:
            futures = {
                pool.submit(self._score_one, run, requirements, c, session): c for c in remote
            }
            for future in as_completed(futures):
                candidate = futures[future]
                results[candidate.path] = future.result()  # re-raises an unreachable provider
                done += 1
                if progress:
                    progress(f"scored {done}/{len(chosen)}: {candidate.name}")
        for candidate in chosen:
            if candidate.locked:
                if progress:
                    progress(f"scoring locally (private): {candidate.name}")
                results[candidate.path] = self._score_one(run, requirements, candidate, session)
        return [results[c.path] for c in chosen]

    def _score_one(
        self,
        run: MatchRun,
        requirements: list[Requirement],
        candidate: MatchCandidate,
        session: bool,
    ) -> DocumentScore:
        settings = self._ctx.settings
        try:
            document = self._loader.load(candidate.path)
            judgement = judge.judge_document(
                self._gateway,
                requirements,
                run.jd_text,
                name=candidate.name,
                document_text=document.text,
                match=settings.match,
                chat=settings.chat,
                session=session,
                local_only=candidate.locked,
            )
        except ProviderUnavailableError:
            raise
        except (MatchError, ProviderError, DocumentError, ChatBlockedError) as exc:
            # one blocked document (never-send, over budget) must not sink the rest of the run
            logger.warning("match: scoring %s failed: %s", candidate.name, exc)
            return DocumentScore(candidate, error=str(exc))
        breakdown = compute_score(requirements, judgement.rows, settings.match)
        return DocumentScore(candidate, breakdown, judgement)

    # ------------------------------------------------------------------ step 4
    def verdict_messages(self, run: MatchRun) -> list[Message]:
        """The verdict prompt: only the step-3 results, never the documents themselves."""
        by_id = {r.id: r.text for r in run.requirements}
        ranking: list[dict[str, object]] = []
        for item in run.ranked():
            if item.breakdown is None or item.judgement is None:
                ranking.append({"file": item.candidate.name, "error": item.error})
                continue
            missing = [
                by_id.get(r.requirement_id, "")
                for r in item.breakdown.rows
                if r.status in ("missing", "unverified")
            ]
            ranking.append(
                {
                    "file": item.candidate.name,
                    "score": item.breakdown.score,
                    "summary": item.breakdown.summary_line,
                    "missing_or_unverified": missing,
                    "seniority_fit": item.judgement.seniority_fit,
                    "notes": item.judgement.summary,
                }
            )
        job, _ = fit_to_budget(run.jd_text, _JD_SUMMARY_TOKENS)
        payload = json.dumps({"job": job, "ranking": ranking}, ensure_ascii=False)
        fence = fence_for(payload)  # the job text and the model's notes are untrusted too
        return [
            Message("system", f"{VERDICT_SYSTEM}\n\n{fence.rule}"),
            Message("user", fence.wrap(payload)),
        ]

    def stream_verdict(self, run: MatchRun, session: bool = False) -> Iterator[str]:
        local_only = any(s.candidate.locked for s in run.scores)
        for chunk in self._gateway.stream(
            self.verdict_messages(run), ROLE_CHAT, session=session, local_only=local_only
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------ step 5
    def chat_context(self, run: MatchRun, top: int | None = None) -> tuple[list[str], str]:
        """Pinned file paths and scratch text for the follow-up chat.

        Pinned: the scored documents in full. Scratch: the job description and their step-3
        results, so "what's missing?" and "rewrite my bullets" start with everything known.
        """
        ranked = [s for s in run.ranked() if s.breakdown is not None and s.judgement is not None]
        chosen = ranked[:top] if top else ranked
        results = []
        for item in chosen:
            assert item.breakdown is not None and item.judgement is not None
            head = f"{item.candidate.name}: score {item.score}/100; {item.breakdown.summary_line}"
            detail = judge.judgement_json(run.requirements, item.judgement)
            results.append(f"{head}\n{detail}")
        scratch = f"JOB DESCRIPTION:\n{run.jd_text}\n\nSCORING RESULTS:\n" + "\n\n".join(results)
        return [s.candidate.path for s in chosen], scratch
