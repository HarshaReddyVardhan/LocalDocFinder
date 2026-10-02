import json
import re
from typing import Any

import pytest
from tests.core.conftest import Chat, Env

from vector_embed.core.documents import DocumentLoader
from vector_embed.core.match.judge import MatchError
from vector_embed.core.match.pipeline import MatchPipeline
from vector_embed.core.match.recall import select_all, select_none
from vector_embed.core.providers.base import ProviderUnavailableError
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.chat import ChatInput, ChatSkill

JD = (
    "Senior backend engineer wanted. Requirements: Python, PostgreSQL. "
    "Nice to have: Kubernetes, AWS. Five years of experience with payment systems."
)
CHECKLIST = {
    "requirements": [
        {"id": 1, "text": "Python", "kind": "must", "category": "skill"},
        {"id": 2, "text": "PostgreSQL", "kind": "must", "category": "skill"},
        {"id": 3, "text": "Kubernetes", "kind": "nice", "category": "skill"},
        {"id": 4, "text": "AWS", "kind": "nice", "category": "skill"},
    ]
}
KEYWORDS = {1: "python", 2: "postgresql", 3: "kubernetes", 4: "aws"}


def resume(name: str, *skills: str, extra: str = "") -> str:
    return (
        f"{name}\nSummary\nBackend engineer building payment systems\nWork Experience\n"
        f"Built payment APIs using {', '.join(skills)}\n{extra}Education\nBSc Computer Science\n"
        f"Skills\n{', '.join(skills)}\nProjects\nSearch engine\n"
    )


def faithful_model(kwargs: dict[str, Any]) -> str:
    """A scripted 'LLM': extracts the fixed checklist, and judges by keyword with real quotes."""
    system = kwargs["messages"][0]["content"]
    if system.startswith("You extract"):
        return json.dumps(CHECKLIST)
    user = kwargs["messages"][-1]["content"]
    document = user.split("Resume (", 1)[1].split("):\n", 1)[1].lower()
    rows = []
    for rid, word in KEYWORDS.items():
        if word in document:
            start = document.index(word)
            quote = document[max(0, start - 5) : start + len(word) + 5]
            rows.append({"id": rid, "status": "met", "evidence_quote": quote})
        else:
            rows.append({"id": rid, "status": "missing", "evidence_quote": ""})
    return json.dumps({"results": rows, "seniority_fit": "fit", "summary": "Reasoned."})


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def library(env: Env, chat: Chat) -> dict[str, str]:
    chat.client.chat_json_fn = faithful_model
    paths = {
        "v1": write(
            env, "Resume_v1.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes", "AWS")
        ),
        "v2": write(
            env, "Resume_v2.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes", "AWS")
        ),
        "python": write(env, "Resume_python.txt", resume("Sam Roe", "Python")),
        "java": write(env, "Resume_java.txt", resume("Joe Bloggs", "Java", "Spring")),
    }
    env.indexer.index_paths(list(paths.values()))
    env.indexer.assign_version_groups()
    env.store.maintain()
    return paths


@pytest.fixture
def pipeline(skill_ctx: SkillContext, chat: Chat, env: Env) -> MatchPipeline:
    loader = DocumentLoader(env.store, env.scope, lambda: env.extractors)
    return MatchPipeline(skill_ctx, chat.gateway, loader)


def requests(chat: Chat) -> list[str]:
    return [json.dumps(c["messages"]) for c in chat.chat_calls()]


class TestWorkflow:
    def test_ranks_the_best_resume_first_and_shows_only_the_newest_version(
        self, pipeline: MatchPipeline, library: dict[str, str]
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        assert [c.name for c in run.candidates].count("Resume_v1.txt") + [
            c.name for c in run.candidates
        ].count("Resume_v2.txt") == 1
        pipeline.score(run)
        ranked = run.ranked()
        assert ranked[0].candidate.name in {"Resume_v1.txt", "Resume_v2.txt"}
        assert ranked[0].score == 100
        assert [s.score for s in ranked] == sorted([s.score for s in ranked], reverse=True)
        assert ranked[-1].candidate.name == "Resume_java.txt"
        assert ranked[-1].score == 0

    def test_the_checklist_is_generated_once_and_reused_for_every_document(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        extractions = [r for r in requests(chat) if "You extract a hiring checklist" in r]
        judgements = [r for r in requests(chat) if "You check ONE resume" in r]
        assert len(extractions) == 1
        assert len(judgements) == len(run.candidates)
        assert all("1. [must] Python" in j for j in judgements)

    def test_only_ticked_documents_are_ever_sent(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        run = pipeline.start(JD, include_old_versions=True)
        select_all(run.candidates)
        run.candidates[0].selected = False
        run.candidates[1].selected = False
        unticked = {c.name for c in run.candidates if not c.selected}
        pipeline.score(run)
        sent = "\n".join(requests(chat))
        for candidate in run.candidates:
            marker = f"Resume ({candidate.name})"
            assert (marker in sent) == candidate.selected
        assert len(unticked) == 2
        assert len(chat.chat_calls()) == 1 + len([c for c in run.candidates if c.selected])

    def test_scores_are_reproducible(
        self, pipeline: MatchPipeline, library: dict[str, str]
    ) -> None:
        first = pipeline.start(JD)
        select_all(first.candidates)
        pipeline.score(first)
        second = pipeline.start(JD)
        select_all(second.candidates)
        pipeline.score(second)
        assert [(s.candidate.name, s.score) for s in first.ranked()] == [
            (s.candidate.name, s.score) for s in second.ranked()
        ]

    def test_every_scoring_call_fits_the_context_window(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str], skill_ctx: SkillContext
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        window = skill_ctx.settings.chat.num_ctx
        for item in run.scores:
            assert item.judgement is not None
            assert (
                item.judgement.prompt_tokens + skill_ctx.settings.match.reserved_output_tokens
                <= window
            )

    def test_a_skill_only_on_the_last_page_is_still_found(
        self, env: Env, chat: Chat, pipeline: MatchPipeline
    ) -> None:
        chat.client.chat_json_fn = faithful_model
        filler = "".join(f"Earlier role {i} with various duties.\n\n" for i in range(60))
        path = write(
            env,
            "Resume_long.txt",
            resume("Ann Lee", "Python", extra=filler + "Kubernetes certified.\n"),
        )
        env.indexer.index_paths([path])
        env.store.maintain()
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        rows = {r.requirement_id: r.status for r in run.scores[0].breakdown.rows}  # type: ignore[union-attr]
        assert rows[3] == "met"

    def test_fabricated_evidence_is_downgraded_to_unverified(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        def liar(kwargs: dict[str, Any]) -> str:
            if kwargs["messages"][0]["content"].startswith("You extract"):
                return json.dumps(CHECKLIST)
            rows = [
                {
                    "id": rid,
                    "status": "met",
                    "evidence_quote": "Led a team of fifty engineers at Google",
                }
                for rid in KEYWORDS
            ]
            return json.dumps({"results": rows, "seniority_fit": "over", "summary": "x"})

        chat.client.chat_json_fn = liar
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        for item in run.scores:
            assert item.breakdown is not None
            assert item.breakdown.score == 0
            assert item.breakdown.unverified == 4
            assert "unverified" in item.breakdown.summary_line

    def test_no_selection_is_an_error(
        self, pipeline: MatchPipeline, library: dict[str, str]
    ) -> None:
        run = pipeline.start(JD)
        select_none(run.candidates)
        with pytest.raises(MatchError, match="no documents are selected"):
            pipeline.score(run)

    def test_user_edits_to_the_checklist_change_the_score(
        self, pipeline: MatchPipeline, library: dict[str, str]
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        checklist = pipeline.build_checklist(run)
        checklist[0].enabled = False  # "ignore the Python requirement"
        checklist[1] = checklist[1].model_copy(update={"weight": 3.0})
        run.requirements = checklist
        pipeline.score(run)
        java = next(s for s in run.scores if s.candidate.name == "Resume_java.txt")
        assert java.breakdown is not None
        assert java.breakdown.total_must == 1  # the unticked requirement is not counted

    def test_progress_is_reported(self, pipeline: MatchPipeline, library: dict[str, str]) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        seen: list[str] = []
        pipeline.score(run, progress=seen.append)
        assert seen[0].startswith("scoring 1/")
        assert len(seen) == len(run.candidates)


class TestFailures:
    def test_one_bad_document_does_not_stop_the_others(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        def flaky(kwargs: dict[str, Any]) -> str:
            user = kwargs["messages"][-1]["content"]
            if "Resume (Resume_java.txt)" in user:
                return '{"results": "garbage"}'
            return faithful_model(kwargs)

        chat.client.chat_json_fn = flaky
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        failed = [s for s in run.scores if s.breakdown is None]
        assert [s.candidate.name for s in failed] == ["Resume_java.txt"]
        assert failed[0].error
        assert run.ranked()[-1] is failed[0]  # errors sort last

    def test_an_unreachable_server_aborts_the_run(
        self, pipeline: MatchPipeline, library: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.build_checklist(run)

        def down(*_a: object, **_k: object) -> None:
            raise ProviderUnavailableError("ollama unavailable")

        monkeypatch.setattr(pipeline._gateway, "chat_json", down)
        with pytest.raises(ProviderUnavailableError):
            pipeline.score(run)

    def test_locked_documents_are_judged_locally_only(
        self,
        pipeline: MatchPipeline,
        chat: Chat,
        library: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run = pipeline.start(JD)
        select_all(run.candidates)
        run.candidates[0].locked = True
        seen: list[bool] = []
        original = pipeline._gateway.chat_json

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("local_only", False))
            return original(*args, **kwargs)

        monkeypatch.setattr(pipeline._gateway, "chat_json", spy)
        pipeline.score(run)
        assert seen[0] is False  # the checklist call sends only the job description
        assert sorted(seen[1:]) == [False] * (len(run.candidates) - 1) + [True]


class TestVerdictAndChat:
    def scored(self, pipeline: MatchPipeline) -> Any:
        run = pipeline.start(JD)
        select_all(run.candidates)
        pipeline.score(run)
        return run

    def test_the_verdict_sees_only_the_step_three_json_never_the_documents(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        run = self.scored(pipeline)
        messages = pipeline.verdict_messages(run)
        payload = json.loads(messages[1].content)
        assert next(r["file"] for r in payload["ranking"]) in {"Resume_v1.txt", "Resume_v2.txt"}
        assert payload["ranking"][0]["score"] == 100
        assert "Kubernetes" in payload["ranking"][-1]["missing_or_unverified"]
        assert "Built payment APIs" not in messages[1].content
        chat.client.chat_reply = ["Jane fits best.", " Joe lacks Python."]
        assert "".join(pipeline.stream_verdict(run)) == "Jane fits best. Joe lacks Python."

    def test_errors_appear_in_the_verdict_payload(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str]
    ) -> None:
        run = self.scored(pipeline)
        run.scores[0].breakdown = None
        run.scores[0].judgement = None
        run.scores[0].error = "boom"
        payload = json.loads(pipeline.verdict_messages(run)[1].content)
        assert {"file": run.scores[0].candidate.name, "error": "boom"} in payload["ranking"]

    def test_follow_up_chat_starts_with_the_jd_the_resumes_and_the_scores(
        self, pipeline: MatchPipeline, chat: Chat, library: dict[str, str], skill_ctx: SkillContext
    ) -> None:
        run = self.scored(pipeline)
        pinned, scratch = pipeline.chat_context(run, top=1)
        assert len(pinned) == 1
        assert "JOB DESCRIPTION" in scratch
        assert "Kubernetes" in scratch
        assert '"status": "met"' in scratch
        skill = ChatSkill(skill_ctx)
        session = skill.open_session("match follow-up", pinned, scratch)
        chat.client.chat_reply = ["You are missing nothing."]
        skill.run(ChatInput(message="What is missing?", session=session))
        system = chat.chat_calls()[-1]["messages"][0]["content"]  # type: ignore[index]
        assert "Built payment APIs" in system  # the resume in full
        assert "SCORING RESULTS" in system
        assert "senior backend engineer" in system.lower()

    def test_chat_context_without_a_limit_uses_every_scored_document(
        self, pipeline: MatchPipeline, library: dict[str, str]
    ) -> None:
        run = self.scored(pipeline)
        pinned, _ = pipeline.chat_context(run)
        assert len(pinned) == len(run.scores)


def test_json_reply_helper_matches_the_schema() -> None:
    reply = json.loads(faithful_model({"messages": [{"content": "You extract..."}]}))
    assert [r["id"] for r in reply["requirements"]] == [1, 2, 3, 4]
    assert re.fullmatch(r"[a-z]+", KEYWORDS[1])
