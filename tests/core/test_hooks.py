import logging
from pathlib import Path

import pytest
from tests.core.conftest import Chat, Env

from localdoc_finder.core import hooks
from localdoc_finder.core.hooks import (
    Answered,
    DocumentClassified,
    FileIndexed,
    HookBus,
    QueryRan,
)
from localdoc_finder.core.skills.ask import AskInput, AskSkill
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.skills.chat import ChatInput, ChatSkill
from localdoc_finder.core.skills.search import SearchSkill

NOTES = "# Decisions\n\nWe retry failed payments with exponential backoff and a jitter.\n"


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


class TestBus:
    def test_handlers_get_only_their_event_type(self) -> None:
        bus = HookBus()
        seen: list[object] = []
        bus.subscribe(QueryRan, seen.append)
        bus.emit(QueryRan("q", 3))
        bus.emit(FileIndexed("a.txt", 2))
        assert seen == [QueryRan("q", 3)]

    def test_unsubscribe_and_scoped_subscription(self) -> None:
        bus = HookBus()
        seen: list[object] = []
        unsubscribe = bus.subscribe(QueryRan, seen.append)
        unsubscribe()
        unsubscribe()  # twice is harmless
        with bus.subscribed(QueryRan, seen.append):
            bus.emit(QueryRan("inside", 0))
        bus.emit(QueryRan("after", 0))
        assert seen == [QueryRan("inside", 0)]

    def test_a_failing_handler_is_logged_and_the_others_still_run(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus = HookBus()
        seen: list[object] = []

        def broken(_event: QueryRan) -> None:
            raise RuntimeError("boom")

        bus.subscribe(QueryRan, broken)
        bus.subscribe(QueryRan, seen.append)
        with caplog.at_level(logging.ERROR, logger=hooks.__name__):
            bus.emit(QueryRan("q", 1))
        assert seen == [QueryRan("q", 1)]
        assert "hook failed" in caplog.text

    def test_the_decorator_subscribes_to_the_process_bus(self) -> None:
        seen: list[object] = []

        @hooks.on(Answered)
        def record(event: Answered) -> None:
            seen.append(event)

        try:
            hooks.emit(Answered("ask", "q", "a"))
        finally:
            hooks.BUS._handlers[Answered].remove(record)  # type: ignore[arg-type]
        assert seen == [Answered("ask", "q", "a")]


class TestEmitters:
    def test_indexing_announces_files_and_their_types(self, env: Env) -> None:
        path = write(env, "decisions.md", NOTES)
        indexed: list[FileIndexed] = []
        classified: list[DocumentClassified] = []
        with (
            hooks.BUS.subscribed(FileIndexed, indexed.append),
            hooks.BUS.subscribed(DocumentClassified, classified.append),
        ):
            env.indexer.index_paths([path])
        assert [(e.path, e.chunks > 0) for e in indexed] == [(path, True)]
        assert [(e.path, e.title) for e in classified] == [(path, "# Decisions")]
        assert classified[0].doc_type

    def test_search_announces_the_query(self, env: Env, skill_ctx: SkillContext) -> None:
        env.indexer.index_paths([write(env, "decisions.md", NOTES)])
        env.store.maintain()
        seen: list[QueryRan] = []
        with hooks.BUS.subscribed(QueryRan, seen.append):
            results = SearchSkill(skill_ctx).search("retry payments")
        assert seen == [QueryRan("retry payments", len(results))]

    def test_ask_and_chat_announce_their_answers(
        self, env: Env, skill_ctx: SkillContext, chat: Chat
    ) -> None:
        path = write(env, "decisions.md", NOTES)
        env.indexer.index_paths([path])
        env.store.maintain()
        chat.client.chat_reply = ["We back off [1]."]
        seen: list[Answered] = []
        with hooks.BUS.subscribed(Answered, seen.append):
            AskSkill(skill_ctx).run(AskInput(question="how do we retry failed payments"))
            chat.client.chat_reply = ["Sure."]
            ChatSkill(skill_ctx).run(ChatInput(message="hello", pin=[path]))
        ask, chatted = seen
        assert (ask.skill, ask.question, ask.answer) == (
            "ask",
            "how do we retry failed payments",
            "We back off [1].",
        )
        assert [Path(p).name for p in ask.sources] == ["decisions.md"]
        assert (chatted.skill, chatted.question, chatted.answer) == ("chat", "hello", "Sure.")
