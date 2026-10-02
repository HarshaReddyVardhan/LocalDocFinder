from pathlib import Path

import pytest
from tests.core.conftest import Chat, Env, Power

from vector_embed.core.llm import ChatBlockedError
from vector_embed.core.rag import NOT_FOUND
from vector_embed.core.skills.ask import AskInput, AskResult, AskSkill, gateway_of
from vector_embed.core.skills.base import SkillContext, create_skill

PAYMENTS = (
    "def retry_failed_payments(order):\n"
    "    # retry the charge with exponential backoff\n"
    "    return charge_card(order)\n"
)
NOTES = "# Decisions\n\nWe retry failed payments with exponential backoff and a jitter.\n"


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def indexed(env: Env, skill_ctx: SkillContext, chat: Chat) -> dict[str, str]:
    paths = {
        "code": write(env, "payments.py", PAYMENTS),
        "notes": write(env, "decisions.md", NOTES),
    }
    env.indexer.index_paths(list(paths.values()))
    env.store.maintain()
    return paths


@pytest.fixture
def ask(skill_ctx: SkillContext) -> AskSkill:
    return AskSkill(skill_ctx)


def prompt_of(chat: Chat) -> str:
    return str(chat.chat_calls()[0]["messages"][1]["content"])  # type: ignore[index]


class TestAnswers:
    def test_streams_the_answer_with_checked_citations(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["Payments retry ", "with backoff [1]."]
        run = ask.prepare("how do we retry failed payments")
        text = "".join(run.deltas())
        assert text == "Payments retry with backoff [1]."
        assert [s.n for s in run.result.cited] == [1]
        footer = run.footer()
        assert footer.startswith("\n\nSources:\n[1] ")
        assert Path(run.result.cited[0].path).name in footer
        assert ":" in footer.rsplit("\n", 1)[-1]  # path:line form

    def test_prompt_contains_only_retrieved_sources(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        ask.run(AskInput(question="how do we retry failed payments"))
        prompt = prompt_of(chat)
        assert "exponential backoff" in prompt
        assert "Question: how do we retry failed payments" in prompt

    def test_unknown_question_is_not_found_without_calling_the_model(
        self, ask: AskSkill, chat: Chat, env: Env, skill_ctx: SkillContext
    ) -> None:
        run = ask.prepare("anything at all")
        assert "".join(run.deltas()) == NOT_FOUND
        assert run.result.not_found
        assert run.footer() == ""
        assert chat.chat_calls() == []

    def test_model_saying_not_found_is_respected(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = [NOT_FOUND]
        result = ask.run(AskInput(question="retry failed payments"))
        assert result.not_found
        assert result.cited == []

    def test_invented_citations_are_flagged_and_dropped(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["It retries [1] and logs [42]."]
        run = ask.prepare("retry failed payments")
        list(run.deltas())
        assert run.result.invalid_citations == {42}
        assert [s.n for s in run.result.cited] == [1]
        assert "[42]" in run.footer()

    def test_uncited_answers_get_a_warning_and_the_searched_sources(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["It uses backoff."]
        run = ask.prepare("retry failed payments")
        list(run.deltas())
        footer = run.footer()
        assert "cited no sources" in footer
        assert "Searched:" in footer

    def test_search_filters_work_inside_a_question(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        ask.run(AskInput(question="retry failed payments ext:md"))
        prompt = prompt_of(chat)
        assert "decisions.md" in prompt
        assert "payments.py" not in prompt


class TestGpuAndRouting:
    def test_query_embeds_on_cpu_and_the_embedder_is_unloaded_before_chat(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str], skill_ctx: SkillContext
    ) -> None:
        seen: list[bool] = []
        original = skill_ctx.embedder.embed

        def spy(texts: list[str], kind: str = "doc", cpu: bool = False):  # type: ignore[no-untyped-def]
            seen.append(cpu)
            return original(texts, kind, cpu)  # type: ignore[arg-type]

        skill_ctx.embedder.embed = spy  # type: ignore[method-assign]
        ask.run(AskInput(question="retry failed payments"))
        assert seen == [True]
        kinds = chat.kinds()
        assert kinds.index("embed") < kinds.index("chat")  # unload request precedes the LLM
        assert chat.client.calls[kinds.index("embed")][1]["keep_alive"] == 0

    def test_code_heavy_context_goes_to_the_code_model(
        self, ask: AskSkill, chat: Chat, env: Env, skill_ctx: SkillContext
    ) -> None:
        env.indexer.index_paths([write(env, "payments.py", PAYMENTS)])
        env.store.maintain()
        result = ask.run(AskInput(question="retry charge exponential backoff"))
        assert result.role == "code_chat"
        assert chat.chat_calls()[0]["model"] == "qwen2.5-coder:7b"

    def test_prose_goes_to_the_general_model(
        self, ask: AskSkill, chat: Chat, env: Env, skill_ctx: SkillContext
    ) -> None:
        env.indexer.index_paths([write(env, "decisions.md", NOTES)])
        env.store.maintain()
        result = ask.run(AskInput(question="retry failed payments jitter"))
        assert result.role == "chat"
        assert chat.chat_calls()[0]["model"] == "qwen3.5:9b"

    def test_code_routing_can_be_disabled(
        self, ask: AskSkill, chat: Chat, env: Env, skill_ctx: SkillContext
    ) -> None:
        skill_ctx.settings = skill_ctx.settings.model_copy(
            update={"chat": skill_ctx.settings.chat.model_copy(update={"code_routing": False})}
        )
        env.indexer.index_paths([write(env, "payments.py", PAYMENTS)])
        env.store.maintain()
        assert ask.run(AskInput(question="retry charge backoff")).role == "chat"

    def test_one_shot_asks_unload_the_model_and_sessions_keep_it(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        ask.run(AskInput(question="retry failed payments"))
        ask.run(AskInput(question="retry failed payments", session=True))
        assert [c["keep_alive"] for c in chat.chat_calls()] == [0, "10m"]

    def test_battery_blocks_local_answers(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str], power_state: Power
    ) -> None:
        power_state.on_ac = False
        with pytest.raises(ChatBlockedError):
            ask.run(AskInput(question="retry failed payments"))


class TestSkillInterface:
    def test_stream_run_and_render(
        self, ask: AskSkill, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["Answer [1]."]
        streamed = "".join(ask.stream(AskInput(question="retry failed payments")))
        assert streamed.startswith("Answer [1].")
        assert "Sources:" in streamed
        result = ask.run(AskInput(question="retry failed payments"))
        assert isinstance(result, AskResult)
        assert ask.render(result) == "Answer [1]."

    def test_registered_for_the_cli_and_ui(self, skill_ctx: SkillContext, chat: Chat) -> None:
        skill = create_skill("ask", skill_ctx)
        assert isinstance(skill, AskSkill)
        assert skill.ui_hint == "panel"
        assert "chat" in skill.roles

    def test_missing_gateway_is_a_clear_error(self, skill_ctx: SkillContext) -> None:
        skill_ctx.extras.pop("llm", None)
        with pytest.raises(RuntimeError, match="no LLM gateway"):
            gateway_of(skill_ctx)
