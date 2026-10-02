from pathlib import Path

import pytest
from tests.core.conftest import Chat, Env, Power

from vector_embed.app.assistant import (
    AssistantService,
    ChatState,
    Delta,
    Failed,
    Finished,
)
from vector_embed.core.skills.base import SkillContext

NOTES = "# Decisions\n\nWe retry failed payments with exponential backoff.\n"


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def service(skill_ctx: SkillContext, chat: Chat) -> AssistantService:
    return AssistantService(lambda: skill_ctx)


def drain(events: object) -> tuple[str, list[object]]:
    text, others = "", []
    for event in events:  # type: ignore[attr-defined]
        if isinstance(event, Delta):
            text += event.text
        else:
            others.append(event)
    return text, others


class TestAsk:
    def test_streams_deltas_then_finishes_with_cited_sources_first(
        self, service: AssistantService, chat: Chat, env: Env
    ) -> None:
        env.indexer.index_paths([write(env, "decisions.md", NOTES)])
        env.store.maintain()
        chat.client.chat_reply = ["We use ", "backoff [1]."]
        text, (finished,) = drain(service.ask("how do we retry failed payments"))
        assert text == "We use backoff [1]."
        assert isinstance(finished, Finished)
        assert finished.sources[0].n == 1
        assert "Sources:" in finished.note

    def test_errors_become_a_failed_event(
        self, service: AssistantService, chat: Chat, env: Env, power_state: Power
    ) -> None:
        env.indexer.index_paths([write(env, "decisions.md", NOTES)])
        env.store.maintain()
        power_state.on_ac = False
        _, (failed,) = drain(service.ask("retry failed payments"))
        assert isinstance(failed, Failed)
        assert "plug in" in failed.message

    def test_nothing_indexed_is_a_not_found_answer(self, service: AssistantService) -> None:
        text, (finished,) = drain(service.ask("anything"))
        assert "Not found" in text
        assert isinstance(finished, Finished)


class TestChat:
    def test_first_message_creates_the_session_with_pins_and_scratch(
        self, service: AssistantService, chat: Chat, env: Env
    ) -> None:
        state = ChatState(pinned=[write(env, "r.txt", "resume text")], scratch="job description")
        text, (finished,) = drain(service.chat("what is missing?", state))
        assert text == "Hello"
        assert state.session_id == finished.session_id  # type: ignore[union-attr]
        system = chat.chat_calls()[0]["messages"][0]["content"]  # type: ignore[index]
        assert "resume text" in system
        assert "job description" in system

    def test_follow_ups_reuse_the_session(self, service: AssistantService, chat: Chat) -> None:
        state = ChatState(scratch="jd")
        drain(service.chat("first", state))
        first = state.session_id
        drain(service.chat("second", state))
        assert state.session_id == first
        roles = [m["role"] for m in chat.chat_calls()[1]["messages"]]  # type: ignore[index]
        assert roles == ["system", "user", "assistant", "user"]

    def test_pinning_a_secret_fails_cleanly(self, service: AssistantService, env: Env) -> None:
        state = ChatState(pinned=[write(env, ".env", "TOKEN=1")])
        _, (failed,) = drain(service.chat("hi", state))
        assert isinstance(failed, Failed)
        assert "secret" in failed.message
        assert state.session_id is None

    def test_truncation_is_reported_in_the_final_event(
        self, service: AssistantService, env: Env
    ) -> None:
        big = write(env, "big.txt", "many words in a long document\n" * 4000)
        _, (finished,) = drain(service.chat("summarise", ChatState(pinned=[big])))
        assert "cut to fit the context" in finished.note  # type: ignore[union-attr]


class TestLifecycle:
    def test_begin_prewarms_and_end_unloads(self, service: AssistantService, chat: Chat) -> None:
        service.begin_chat()
        assert service.session_active
        generate = [kw for kind, kw in chat.client.calls if kind == "generate"]
        assert generate[0]["keep_alive"] == "10m"
        service.begin_chat()  # idempotent
        service.end_chat()
        assert not service.session_active
        assert [kw["keep_alive"] for kind, kw in chat.client.calls if kind == "generate"][-1] == 0

    def test_end_and_maintain_before_anything_was_built_are_noops(
        self, skill_ctx: SkillContext
    ) -> None:
        built: list[int] = []

        def factory() -> SkillContext:
            built.append(1)
            return skill_ctx

        service = AssistantService(factory)
        service.end_chat()
        assert service.maintain() is None
        assert built == []

    def test_maintain_reports_why_the_model_was_unloaded(
        self, service: AssistantService, chat: Chat, power_state: Power
    ) -> None:
        service.begin_chat()
        power_state.on_ac = False
        assert service.maintain() == "unplugged"
        assert not service.session_active


def test_chat_state_helpers(tmp_path: Path) -> None:
    state = ChatState(session_id=3, pinned=["a", "b"], scratch="xyz")
    assert state.describe() == "2 file(s) pinned, pasted text (3 chars)"
    state.reset()
    assert (state.session_id, state.pinned, state.scratch) == (None, [], "")
    assert state.describe() == ""
