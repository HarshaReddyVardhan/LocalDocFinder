"""LLM steps of Match: build the requirement checklist once, then judge one document at a time.

Every judging call sends the same fixed checklist plus the *whole* text of ONE document, so
"missing" really means missing (no chunk the model never saw) and results are comparable. If a
document does not fit the context budget it is cut down to the sections most relevant to the
job description, and that is reported (``reduced``) as reduced reliability.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from vector_embed.core.llm import LlmGateway
from vector_embed.core.match.scoring import Requirement, RowResult, verify_rows
from vector_embed.core.models.catalog import ROLE_MATCH_SCORER
from vector_embed.core.prompt_safety import fence_for
from vector_embed.core.providers.base import (
    InvalidJsonError,
    Message,
    ProviderError,
    ProviderUnavailableError,
)
from vector_embed.core.relevance import keywords, reduce_to_relevant
from vector_embed.core.settings import ChatSettings, MatchSettings
from vector_embed.core.tokens import estimate_tokens, fit_to_budget

logger = logging.getLogger(__name__)

_JD_SUMMARY_TOKENS = 400
_MIN_JD_TOKENS = 200  # never cut a job description to less than this, however small the window
_FENCE_TOKENS = 20  # the fence lines around the job description
_JUDGE_ATTEMPTS = 2  # a malformed judgement is usually fine on a second try

REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "kind": {"enum": ["must", "nice"]},
                    "category": {"enum": ["skill", "experience", "education", "cert", "other"]},
                    "years": {"type": ["integer", "null"]},
                },
                "required": ["id", "text", "kind", "category"],
            },
        }
    },
    "required": ["requirements"],
}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "status": {"enum": ["met", "partial", "missing"]},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["id", "status", "evidence_quote"],
            },
        },
        "seniority_fit": {"enum": ["under", "fit", "over"]},
        "summary": {"type": "string"},
    },
    "required": ["results", "seniority_fit", "summary"],
}

EXTRACT_SYSTEM = (
    "You extract a hiring checklist from a job description. Reply with JSON only. List every "
    "distinct requirement: skills, tools, experience, education and certifications. Use kind "
    "'must' for required, minimum or mandatory items and 'nice' for preferred or bonus items. "
    "Keep each text under 12 words. Number ids from 1. Set years to the minimum years of "
    "experience when the text states it, otherwise null. At most {limit} items."
)
JUDGE_SYSTEM = (
    "You check ONE resume against a fixed checklist. For each requirement id decide: 'met' "
    "(clearly shown), 'partial' (related or weaker evidence) or 'missing' (not shown anywhere in "
    "the resume). For met or partial, evidence_quote MUST be an exact short quote copied from the "
    "resume (at most 25 words); for missing leave it empty. seniority_fit is under, fit or over "
    "relative to the role. summary is two sentences. Judge only from the resume text."
)


class MatchError(RuntimeError):
    """The model could not produce a usable checklist or judgement."""


class _RequirementsPayload(BaseModel):
    requirements: list[Requirement]


class _JudgePayload(BaseModel):
    class Row(BaseModel):
        id: int
        status: str
        evidence_quote: str = ""

    results: list[Row]
    seniority_fit: str = "fit"
    summary: str = ""


@dataclass(frozen=True)
class Judgement:
    rows: list[RowResult]
    seniority_fit: str
    summary: str
    reduced: bool
    prompt_tokens: int


def _describe(exc: Exception) -> str:
    """An error for logs and messages. A pydantic error quotes the offending value, which here is
    model output that may repeat the user's document: say where and what, never the input."""
    if isinstance(exc, ValidationError):
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'response'}: {error['type']}"
            for error in exc.errors(include_input=False)
        )
        return f"the model's reply did not match the expected shape ({problems})"
    return str(exc)


def _json(
    gateway: LlmGateway,
    messages: list[Message],
    schema: dict[str, Any],
    session: bool,
    local_only: bool = False,
) -> object:
    return gateway.chat_json(
        messages, schema, ROLE_MATCH_SCORER, session=session, local_only=local_only
    ).data


def requirements_messages(
    jd_text: str, settings: MatchSettings, chat: ChatSettings | None = None
) -> list[Message]:
    """The checklist-extraction prompt (also what the "what will be sent" preview shows).

    A job description longer than the context window is cut to fit: Ollama would otherwise
    silently drop its *start*, which is where the role and its must-haves usually are.
    """
    fence = fence_for(jd_text)
    system = f"{EXTRACT_SYSTEM.format(limit=settings.max_requirements)}\n\n{fence.rule}"
    num_ctx = (chat or ChatSettings()).num_ctx
    room = num_ctx - settings.reserved_output_tokens - estimate_tokens(system)
    body, cut = fit_to_budget(jd_text, max(_MIN_JD_TOKENS, room - _FENCE_TOKENS))
    if cut:
        logger.warning("match: the job description was cut to fit the context window")
    return [Message("system", system), Message("user", f"Job description:\n{fence.wrap(body)}")]


def extract_requirements(
    gateway: LlmGateway,
    jd_text: str,
    settings: MatchSettings,
    session: bool = False,
    chat: ChatSettings | None = None,
) -> list[Requirement]:
    """The checklist, generated once and reused for every document."""
    messages = requirements_messages(jd_text, settings, chat)
    last: Exception | None = None
    for attempt in range(2):
        try:
            data = _json(gateway, messages, REQUIREMENTS_SCHEMA, session)
            parsed = _RequirementsPayload.model_validate(data)
        except (ValidationError, ProviderError) as exc:
            last = exc
            logger.info("match: checklist attempt %d failed: %s", attempt + 1, _describe(exc))
            if isinstance(exc, ProviderUnavailableError):
                break  # retrying a dead server only wastes time
            continue
        items = [r for r in parsed.requirements if r.text.strip()][: settings.max_requirements]
        if items:
            return [r.model_copy(update={"id": i}) for i, r in enumerate(items, 1)]
        last = MatchError("the model returned no requirements")
    raise MatchError(f"could not extract a checklist: {last}")


def _checklist_text(requirements: list[Requirement]) -> str:
    return "\n".join(f"{r.id}. [{r.kind}] {r.text}" for r in requirements if r.enabled)


def reduce_document(
    text: str, requirements: list[Requirement], jd_text: str, budget_tokens: int
) -> tuple[str, bool]:
    """Fit ``text`` to ``budget_tokens`` by keeping the sections most relevant to the job.

    Sections are paragraphs; their original order is preserved. Returns the text and whether
    anything was dropped (reduced reliability).
    """
    wanted = keywords(jd_text, *(req.text for req in requirements))
    return reduce_to_relevant(text, wanted, budget_tokens)


def build_judge_messages(
    requirements: list[Requirement], jd_text: str, name: str, document: str
) -> list[Message]:
    summary, _ = fit_to_budget(jd_text, _JD_SUMMARY_TOKENS)
    fence = fence_for(summary, document)
    user = (
        f"Job summary:\n{fence.wrap(summary)}\n\nChecklist:\n{_checklist_text(requirements)}\n\n"
        f"Resume ({name}):\n{fence.wrap(document)}"
    )
    return [Message("system", f"{JUDGE_SYSTEM}\n\n{fence.rule}"), Message("user", user)]


def prepare_judge_messages(
    requirements: list[Requirement],
    jd_text: str,
    *,
    name: str,
    document_text: str,
    match: MatchSettings,
    chat: ChatSettings,
) -> tuple[list[Message], bool]:
    """The exact judge prompt for one document, and whether the document had to be cut.

    Shared by scoring and by the "what will be sent" preview, so the two can never differ.
    """
    fixed = build_judge_messages(requirements, jd_text, name, "")
    overhead = sum(estimate_tokens(m.content) for m in fixed)
    room = chat.num_ctx - match.reserved_output_tokens - overhead
    budget = max(200, min(match.scoring_budget_tokens, room))
    body, reduced = reduce_document(document_text, requirements, jd_text, budget)
    return build_judge_messages(requirements, jd_text, name, body), reduced


def _judge_payload(
    gateway: LlmGateway, messages: list[Message], name: str, session: bool, local_only: bool
) -> _JudgePayload:
    """The parsed judgement, asking again once when the reply is not the JSON asked for."""
    last: Exception = MatchError("not judged")
    for attempt in range(1, _JUDGE_ATTEMPTS + 1):
        try:
            return _JudgePayload.model_validate(
                _json(gateway, messages, JUDGE_SCHEMA, session, local_only)
            )
        except (ValidationError, InvalidJsonError) as exc:
            last = exc
            logger.info("match: judgement %d of %s unusable: %s", attempt, name, _describe(exc))
    raise MatchError(f"unusable judgement for {name}: {_describe(last)}") from last


def judge_document(
    gateway: LlmGateway,
    requirements: list[Requirement],
    jd_text: str,
    *,
    name: str,
    document_text: str,
    match: MatchSettings,
    chat: ChatSettings,
    session: bool = False,
    local_only: bool = False,
) -> Judgement:
    """Judge one document. The prompt is kept under the context window (logged token count)."""
    messages, reduced = prepare_judge_messages(
        requirements, jd_text, name=name, document_text=document_text, match=match, chat=chat
    )
    prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
    logger.info(
        "match: judging", extra={"doc": name, "prompt_tokens": prompt_tokens, "reduced": reduced}
    )
    payload = _judge_payload(gateway, messages, name, session, local_only)
    allowed = {"met", "partial", "missing"}
    rows = [
        RowResult(r.id, r.status if r.status in allowed else "missing", r.evidence_quote.strip())  # type: ignore[arg-type]
        for r in payload.results
    ]
    return Judgement(
        rows=verify_rows(rows, document_text, match, requirements),
        seniority_fit=payload.seniority_fit,
        summary=payload.summary.strip(),
        reduced=reduced,
        prompt_tokens=prompt_tokens,
    )


def judgement_json(requirements: list[Requirement], judgement: Judgement) -> str:
    """Compact JSON of one judgement (the form later steps and follow-up chats reuse)."""
    by_id = {r.id: r.text for r in requirements}
    return json.dumps(
        {
            "results": [
                {
                    "requirement": by_id.get(row.requirement_id, str(row.requirement_id)),
                    "status": row.status,
                    "evidence": row.evidence,
                }
                for row in judgement.rows
            ],
            "seniority_fit": judgement.seniority_fit,
            "summary": judgement.summary,
        },
        ensure_ascii=False,
    )
