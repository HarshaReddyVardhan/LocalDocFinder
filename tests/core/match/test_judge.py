import json
from typing import Any

import pytest
from tests.core.conftest import Chat

from vector_embed.core.match import judge
from vector_embed.core.match.judge import MatchError
from vector_embed.core.match.scoring import Requirement, RowResult
from vector_embed.core.providers.base import ProviderUnavailableError
from vector_embed.core.settings import ChatSettings, MatchSettings

JD = "Senior backend engineer. Must know Python and PostgreSQL. Kubernetes is a plus."
RESUME = (
    "Jane Doe\n\nWork Experience\nBuilt payment systems in Python and PostgreSQL.\n\n"
    "Skills\nPython, SQL, Docker, Kubernetes"
)
CHECKLIST = {
    "requirements": [
        {"id": 7, "text": "Python", "kind": "must", "category": "skill", "years": None},
        {"id": 9, "text": "PostgreSQL", "kind": "must", "category": "skill"},
        {"id": 11, "text": "Kubernetes", "kind": "nice", "category": "skill"},
    ]
}
MATCH = MatchSettings()
CHAT = ChatSettings()


def reqs() -> list[Requirement]:
    return [
        Requirement(id=1, text="Python", kind="must"),
        Requirement(id=2, text="PostgreSQL", kind="must"),
        Requirement(id=3, text="Kubernetes", kind="nice"),
    ]


def user_prompt(call: dict[str, Any]) -> str:
    return str(call["messages"][-1]["content"])


class TestExtractRequirements:
    def test_ids_are_renumbered_from_one(self, chat: Chat) -> None:
        chat.client.chat_json_reply = json.dumps(CHECKLIST)
        found = judge.extract_requirements(chat.gateway, JD, MATCH)
        assert [(r.id, r.text, r.kind) for r in found] == [
            (1, "Python", "must"),
            (2, "PostgreSQL", "must"),
            (3, "Kubernetes", "nice"),
        ]
        call = chat.chat_calls()[0]
        assert call["format"] == judge.REQUIREMENTS_SCHEMA
        assert user_prompt(call) == JD
        assert "At most 25 items" in call["messages"][0]["content"]  # type: ignore[index]

    def test_blank_items_are_dropped_and_the_list_is_capped(self, chat: Chat) -> None:
        many = [
            {"id": i, "text": f"skill {i}", "kind": "must", "category": "skill"} for i in range(40)
        ]
        many.insert(0, {"id": 99, "text": "  ", "kind": "must", "category": "skill"})
        chat.client.chat_json_reply = json.dumps({"requirements": many})
        found = judge.extract_requirements(chat.gateway, JD, MatchSettings(max_requirements=5))
        assert [r.text for r in found] == [f"skill {i}" for i in range(5)]

    def test_one_bad_reply_is_retried(self, chat: Chat) -> None:
        replies = iter(['{"requirements": "nope"}', json.dumps(CHECKLIST)])
        chat.client.chat_json_fn = lambda _kw: next(replies)
        assert len(judge.extract_requirements(chat.gateway, JD, MATCH)) == 3
        assert len(chat.chat_calls()) == 2

    def test_persistent_garbage_is_an_error(self, chat: Chat) -> None:
        chat.client.chat_json_reply = '{"requirements": []}'
        with pytest.raises(MatchError, match="no requirements"):
            judge.extract_requirements(chat.gateway, JD, MATCH)
        chat.client.chat_json_reply = '{"wrong": 1}'
        with pytest.raises(MatchError, match="could not extract"):
            judge.extract_requirements(chat.gateway, JD, MATCH)

    def test_unreachable_server_is_not_retried(
        self, chat: Chat, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(*_a: object, **_k: object) -> None:
            raise ProviderUnavailableError("ollama unavailable")

        monkeypatch.setattr(chat.gateway, "chat_json", down)
        with pytest.raises(MatchError):
            judge.extract_requirements(chat.gateway, JD, MATCH)


class TestReduceDocument:
    def test_documents_that_fit_are_untouched(self) -> None:
        assert judge.reduce_document(RESUME, reqs(), JD, 5000) == (RESUME, False)

    def test_relevant_sections_survive_in_their_original_order(self) -> None:
        filler = "\n\n".join(
            f"Hobby paragraph number {i} about gardening and painting. " * 6 for i in range(30)
        )
        closing = "Experienced with Kubernetes and PostgreSQL in Python."
        text = f"Intro about nothing.\n\n{filler}\n\n{closing}"
        reduced, was_reduced = judge.reduce_document(text, reqs(), JD, 150)
        assert was_reduced
        assert "Kubernetes and PostgreSQL" in reduced
        assert judge.estimate_tokens(reduced) <= 150
        assert reduced.index("Experienced") > reduced.index("Hobby") if "Hobby" in reduced else True

    def test_one_giant_section_is_cut(self) -> None:
        reduced, was_reduced = judge.reduce_document("word " * 5000, reqs(), JD, 100)
        assert was_reduced
        assert judge.estimate_tokens(reduced) <= 100

    def test_empty_text(self) -> None:
        assert judge.reduce_document("", reqs(), JD, 10) == ("", False)


class TestJudge:
    def reply(self, rows: list[dict[str, Any]], **extra: Any) -> str:
        return json.dumps({"results": rows, "seniority_fit": "fit", "summary": "Solid.", **extra})

    def test_rows_are_verified_against_the_document(self, chat: Chat) -> None:
        chat.client.chat_json_reply = self.reply(
            [
                {"id": 1, "status": "met", "evidence_quote": "Built payment systems in Python"},
                {"id": 2, "status": "met", "evidence_quote": "Ten years running Oracle at NASA"},
                {"id": 3, "status": "missing", "evidence_quote": ""},
            ]
        )
        result = judge.judge_document(
            chat.gateway,
            reqs(),
            JD,
            name="Resume_v2.pdf",
            document_text=RESUME,
            match=MATCH,
            chat=CHAT,
        )
        assert [r.status for r in result.rows] == ["met", "unverified", "missing"]
        assert result.summary == "Solid."
        assert result.seniority_fit == "fit"
        assert not result.reduced

    def test_skill_on_the_last_page_is_seen_because_the_whole_document_is_sent(
        self, chat: Chat
    ) -> None:
        document = "\n\n".join(f"Paragraph {i} about earlier jobs. " * 5 for i in range(20))
        document += "\n\nKubernetes certified administrator since 2021."
        chat.client.chat_json_reply = self.reply(
            [
                {
                    "id": 3,
                    "status": "met",
                    "evidence_quote": "Kubernetes certified administrator since 2021",
                }
            ]
        )
        result = judge.judge_document(
            chat.gateway, reqs(), JD, name="r.pdf", document_text=document, match=MATCH, chat=CHAT
        )
        assert "Kubernetes certified administrator" in user_prompt(chat.chat_calls()[0])
        assert result.rows[0].status == "met"

    def test_prompt_contains_checklist_name_and_jd_summary_but_only_enabled_items(
        self, chat: Chat
    ) -> None:
        chat.client.chat_json_reply = self.reply([])
        checklist = reqs()
        checklist[2] = checklist[2].model_copy(update={"enabled": False})
        judge.judge_document(
            chat.gateway,
            checklist,
            JD,
            name="Resume_v2.pdf",
            document_text=RESUME,
            match=MATCH,
            chat=CHAT,
        )
        prompt = user_prompt(chat.chat_calls()[0])
        assert "1. [must] Python" in prompt
        assert "Kubernetes\n" not in prompt.split("Resume (")[0].split("Checklist:")[1]
        assert "Resume (Resume_v2.pdf)" in prompt
        assert JD in prompt

    def test_every_call_stays_inside_the_context_window(self, chat: Chat) -> None:
        chat.client.chat_json_reply = self.reply([])
        huge = "Section about Python and PostgreSQL work.\n\n" * 3000
        result = judge.judge_document(
            chat.gateway, reqs(), JD, name="huge.pdf", document_text=huge, match=MATCH, chat=CHAT
        )
        assert result.reduced
        assert result.prompt_tokens + MATCH.reserved_output_tokens <= CHAT.num_ctx

    def test_invalid_statuses_become_missing(self, chat: Chat) -> None:
        chat.client.chat_json_reply = self.reply(
            [{"id": 1, "status": "excellent", "evidence_quote": ""}]
        )
        row = judge.judge_document(
            chat.gateway, reqs(), JD, name="r", document_text=RESUME, match=MATCH, chat=CHAT
        ).rows[0]
        assert row.status == "missing"

    def test_malformed_judgements_are_an_error(self, chat: Chat) -> None:
        chat.client.chat_json_reply = '{"results": "oops"}'
        with pytest.raises(MatchError, match="unusable judgement"):
            judge.judge_document(
                chat.gateway, reqs(), JD, name="r", document_text=RESUME, match=MATCH, chat=CHAT
            )

    def test_judgement_json_is_compact_and_readable(self, chat: Chat) -> None:
        chat.client.chat_json_reply = self.reply(
            [{"id": 1, "status": "met", "evidence_quote": "Built payment systems in Python"}]
        )
        result = judge.judge_document(
            chat.gateway, reqs(), JD, name="r", document_text=RESUME, match=MATCH, chat=CHAT
        )
        data = json.loads(judge.judgement_json(reqs(), result))
        assert data["results"][0] == {
            "requirement": "Python",
            "status": "met",
            "evidence": "Built payment systems in Python",
        }
        assert data["summary"] == "Solid."

    def test_unknown_requirement_ids_are_kept_readable(self) -> None:
        j = judge.Judgement([RowResult(99, "missing")], "fit", "", False, 0)
        assert json.loads(judge.judgement_json(reqs(), j))["results"][0]["requirement"] == "99"
