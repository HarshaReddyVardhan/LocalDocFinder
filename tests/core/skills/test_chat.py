from pathlib import Path

import pytest
from tests.core.conftest import Chat, Env

from vector_embed.core.documents import DocumentError
from vector_embed.core.skills.base import SkillContext, create_skill
from vector_embed.core.skills.chat import (
    ChatInput,
    ChatSkill,
    ChatTurn,
    _pinned_block,
    trim_history,
)
from vector_embed.core.store.lance import CHUNKS
from vector_embed.core.store.sqlite import ChatMessage

JD = "Senior backend engineer. Requirements: Python, PostgreSQL, Kubernetes, 5 years."
RESUME = (
    "Jane Doe\nWork Experience\nBuilt payment systems in Python and PostgreSQL\n"
    "Skills\nPython SQL\n"
)


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def skill(skill_ctx: SkillContext, chat: Chat) -> ChatSkill:
    return ChatSkill(skill_ctx)


def messages_of(chat: Chat, call: int = 0) -> list[dict[str, str]]:
    return chat.chat_calls()[call]["messages"]  # type: ignore[return-value]


class TestSessions:
    def test_pinned_files_and_scratch_text_reach_the_prompt_first(
        self, skill: ChatSkill, chat: Chat, env: Env
    ) -> None:
        resume = write(env, "resume.txt", RESUME)
        run = skill.run(ChatInput(message="What is missing?", pin=[resume], scratch=JD))
        system, user = messages_of(chat)
        assert system["role"] == "system"
        assert "payment systems" in system["content"]
        assert "=== pasted text ===" in system["content"]
        assert "Kubernetes" in system["content"]
        assert user == {"role": "user", "content": "What is missing?"}
        assert run.reply == "Hello"

    def test_history_is_stored_and_replayed_on_follow_ups(
        self, skill: ChatSkill, chat: Chat, env: Env
    ) -> None:
        chat.client.chat_reply = ["It lacks Kubernetes."]
        first = skill.run(ChatInput(message="What is missing?", scratch=JD))
        chat.client.chat_reply = ["Add a bullet about clusters."]
        skill.run(ChatInput(message="Rewrite my bullets", session=first.session_id))
        follow_up = messages_of(chat, 1)
        assert [m["role"] for m in follow_up] == ["system", "user", "assistant", "user"]
        assert follow_up[2]["content"] == "It lacks Kubernetes."
        assert "Kubernetes" in follow_up[0]["content"]  # scratch context persists in the session
        stored = skill.ctx.state.messages(first.session_id)
        assert [(m.role, m.content) for m in stored][-2:] == [
            ("user", "Rewrite my bullets"),
            ("assistant", "Add a bullet about clusters."),
        ]

    def test_scratch_text_is_not_written_to_the_index(self, skill: ChatSkill, env: Env) -> None:
        skill.run(ChatInput(message="hi", scratch=JD))
        assert env.store.count(CHUNKS) == 0

    def test_secrets_cannot_be_pinned(self, skill: ChatSkill, env: Env) -> None:
        with pytest.raises(DocumentError, match="secret"):
            skill.open_session("x", [write(env, ".env", "TOKEN=1")])

    def test_unavailable_pinned_files_are_skipped_later(
        self, skill: ChatSkill, chat: Chat, env: Env
    ) -> None:
        resume = Path(write(env, "resume.txt", RESUME))
        session = skill.open_session("t", [str(resume)], "scratch words")
        resume.unlink()
        skill.run(ChatInput(message="hello", session=session))
        assert "scratch words" in messages_of(chat)[0]["content"]

    def test_empty_replies_are_not_stored(self, skill: ChatSkill, chat: Chat) -> None:
        chat.client.chat_reply = []
        turn = skill.run(ChatInput(message="hi", scratch="x"))
        assert turn.reply == ""
        assert skill.ctx.state.messages(turn.session_id) == []


class TestContextBudget:
    def test_long_documents_are_cut_and_reported(
        self, skill: ChatSkill, chat: Chat, env: Env
    ) -> None:
        big = write(env, "big.txt", ("many words in a long document\n") * 4000)
        turn, deltas = skill.turn(ChatInput(message="summarise", pin=[big]))
        list(deltas)
        assert big in turn.truncated
        system = messages_of(chat)[0]["content"]
        assert "document truncated" in system
        assert len(system) < 30_000  # roughly the 5k-token budget, not the 120k-char file

    def test_budget_is_shared_between_documents(self) -> None:
        from vector_embed.core.documents import LoadedDocument

        docs = [LoadedDocument(f"d{i}", "t", "word " * 4000, "", True) for i in range(2)]
        block, cut = _pinned_block(docs, "", 1000)
        assert cut == ["d0", "d1"]
        assert block.count("=== d") == 2
        assert len(block) < 2 * 4000 * 5

    def test_nothing_pinned_gives_an_empty_block(self) -> None:
        assert _pinned_block([], "   ", 1000) == ("", [])

    def test_history_is_trimmed_oldest_first(self) -> None:
        history = [
            ChatMessage("user" if i % 2 == 0 else "assistant", f"m{i} " + "x" * 400, float(i))
            for i in range(10)
        ]
        kept = trim_history(history, 250)
        assert 1 <= len(kept) < 10
        assert kept[-1].content.startswith("m9")
        assert trim_history(history, 1)[0].content.startswith("m9")  # always keeps the latest
        assert trim_history([], 100) == []

    def test_unknown_history_roles_become_user_messages(self) -> None:
        (message,) = trim_history([ChatMessage("tool", "x", 0.0)], 100)
        assert message.role == "user"


class TestFallbackRetrieval:
    def test_without_pins_the_chat_retrieves_from_the_index(
        self, skill: ChatSkill, chat: Chat, env: Env
    ) -> None:
        env.indexer.index_paths(
            [write(env, "notes.md", "# Notes\n\nWe retry failed payments with backoff.\n")]
        )
        env.store.maintain()
        skill.run(ChatInput(message="how do we retry failed payments"))
        assert "retry failed payments" in messages_of(chat)[0]["content"]


class TestSkillInterface:
    def test_stream_appends_the_session_id(self, skill: ChatSkill, chat: Chat) -> None:
        text = "".join(skill.stream(ChatInput(message="hi", scratch="x")))
        assert text.startswith("Hello")
        assert "[session " in text

    def test_stream_reports_truncation(self, skill: ChatSkill, env: Env) -> None:
        big = write(env, "big.txt", ("many words in a long document\n") * 4000)
        assert "cut to fit the context" in "".join(skill.stream(ChatInput(message="x", pin=[big])))

    def test_render_and_registry(self, skill: ChatSkill, skill_ctx: SkillContext) -> None:
        turn = ChatTurn(1, "reply")
        assert skill.render(turn) == "reply"
        assert isinstance(create_skill("chat", skill_ctx), ChatSkill)

    def test_keep_loaded_and_sessions_control_keep_alive(
        self, skill: ChatSkill, chat: Chat
    ) -> None:
        skill.run(ChatInput(message="a", scratch="x"))
        skill.run(ChatInput(message="b", scratch="x", keep_loaded=True))
        chat.gateway.begin_chat()
        skill.run(ChatInput(message="c", scratch="x"))
        assert [c["keep_alive"] for c in chat.chat_calls()] == [0, "10m", "10m"]

    def test_missing_loader_is_a_clear_error(self, skill_ctx: SkillContext, chat: Chat) -> None:
        skill_ctx.extras.pop("documents")
        with pytest.raises(RuntimeError, match="document loader"):
            ChatSkill(skill_ctx)
