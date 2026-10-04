from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.core.conftest import Chat, CloudRig, Env, Power

from localdoc_finder.app.assistant import (
    AssistantService,
    ChatState,
    Delta,
    Failed,
    Finished,
)
from localdoc_finder.core.privacy.policy import Outbound
from localdoc_finder.core.skills.base import SkillContext

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


class TestAnswerBetter:
    @pytest.fixture
    def local_by_default(self, cloud: CloudRig) -> CloudRig:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        cloud.consent.revoke()
        return cloud

    def index(self, env: Env) -> None:
        env.indexer.index_paths(
            [
                write(env, "payments.md", NOTES + "Ticket SSN 123-45-6789 leaked.\n"),
                write(
                    env,
                    ".claude/projects/demo/memory/private.md",
                    "# Private\n\nretry failed payments secretly\n",
                ),
            ]
        )
        env.store.maintain()

    def test_not_available_without_a_cloud_provider(
        self, skill_ctx: SkillContext, chat: Chat
    ) -> None:
        service = AssistantService(lambda: skill_ctx)
        assert not service.cloud_available()
        assert service.cloud_preview_ask("q") is None
        assert service.cloud_preview_chat("q", ChatState()) is None
        _, (failed,) = drain(service.escalated(service.ask("q")))
        assert isinstance(failed, Failed)
        assert "no cloud provider" in failed.message

    def test_the_preview_shows_destination_badge_masking_and_exact_text(
        self, env: Env, skill_ctx: SkillContext, local_by_default: CloudRig
    ) -> None:
        self.index(env)
        service = AssistantService(lambda: skill_ctx)
        assert service.cloud_available()
        preview = service.cloud_preview_ask("how do we retry failed payments")
        assert preview is not None
        assert preview.destination == "OpenRouter / vendor/chat"
        assert preview.badge.startswith("☁ Sending 1 excerpt")
        assert "OpenRouter / vendor/chat" in preview.badge
        assert preview.shield.startswith("🛡 1 sensitive item will be masked")
        assert "[SSN REMOVED]" in preview.text
        assert "123-45-6789" not in preview.text
        assert "secretly" not in preview.text  # private files are filtered like a cloud request
        assert local_by_default.inner.sent == []  # previewing sends nothing
        assert not local_by_default.consent.granted

    def test_escalating_one_request_uses_the_cloud_then_returns_to_local(
        self, env: Env, skill_ctx: SkillContext, chat: Chat, local_by_default: CloudRig
    ) -> None:
        self.index(env)
        service = AssistantService(lambda: skill_ctx)
        local_by_default.inner.reply = ["Cloud answer [1]."]
        text, (finished,) = drain(service.escalated(service.ask("how do we retry failed payments")))
        assert text == "Cloud answer [1]."
        assert isinstance(finished, Finished)
        assert not local_by_default.consent.granted  # consent covered that one request only
        assert local_by_default.inner.sent
        assert chat.chat_calls() == []
        assert not local_by_default.router.escalate  # back to local for the next request
        drain(service.ask("how do we retry failed payments"))
        assert chat.chat_calls()  # now handled by the local model

    def test_the_preview_is_exactly_the_text_that_is_sent(
        self, env: Env, skill_ctx: SkillContext, chat: Chat, local_by_default: CloudRig
    ) -> None:
        self.index(env)
        service = AssistantService(lambda: skill_ctx)
        question = "how do we retry failed payments ext:md"  # a filter the preview must honour
        preview = service.cloud_preview_ask(question)
        assert preview is not None
        local_by_default.inner.reply = ["Cloud answer."]
        drain(service.ask_escalated(question))
        sent = Outbound(local_by_default.inner.sent[-1])  # exactly what reached the provider
        assert preview.text == local_by_default.privacy.preview(sent)
        assert "123-45-6789" not in preview.text

    def test_a_different_question_is_not_sent_with_an_old_preview(
        self, env: Env, skill_ctx: SkillContext, chat: Chat, local_by_default: CloudRig
    ) -> None:
        self.index(env)
        service = AssistantService(lambda: skill_ctx)
        service.cloud_preview_ask("how do we retry failed payments")
        local_by_default.inner.reply = ["ok"]
        drain(service.ask_escalated("what is the refund policy"))
        sent = Outbound(local_by_default.inner.sent[-1])  # exactly what reached the provider
        assert "refund policy" in local_by_default.privacy.preview(sent)

    def test_the_chat_preview_is_what_the_chat_turn_sends(
        self, env: Env, skill_ctx: SkillContext, chat: Chat, local_by_default: CloudRig
    ) -> None:
        self.index(env)
        service = AssistantService(lambda: skill_ctx)
        state = ChatState()
        preview = service.cloud_preview_chat("how do we retry failed payments", state)
        assert preview is not None
        local_by_default.inner.reply = ["Cloud answer."]
        drain(service.chat_escalated("how do we retry failed payments", state))
        sent = Outbound(local_by_default.inner.sent[-1])  # exactly what reached the provider
        assert preview.text == local_by_default.privacy.preview(sent)

    def test_escalation_resets_even_if_the_stream_fails(
        self, env: Env, skill_ctx: SkillContext, local_by_default: CloudRig
    ) -> None:
        service = AssistantService(lambda: skill_ctx)

        def boom() -> Iterator[object]:
            yield Delta("x")
            raise RuntimeError("stream died")

        with pytest.raises(RuntimeError):
            list(service.escalated(boom()))  # type: ignore[arg-type]
        assert not local_by_default.router.escalate
        assert not local_by_default.consent.granted

    def test_chat_preview_opens_the_session_and_includes_pinned_text(
        self, env: Env, skill_ctx: SkillContext, local_by_default: CloudRig
    ) -> None:
        state = ChatState(pinned=[write(env, "r.txt", "Resume of Jane\nPassport No: K1234567")])
        service = AssistantService(lambda: skill_ctx)
        preview = service.cloud_preview_chat("what is missing?", state)
        assert preview is not None
        assert state.session_id is not None
        assert "Resume of Jane" in preview.text
        assert "[PASSPORT REMOVED]" in preview.text
        assert "K1234567" not in preview.text
        assert preview.badge.startswith("☁ Sending 1 excerpt")


class TestConsentLifetime:
    @pytest.fixture
    def local_by_default(self, cloud: CloudRig) -> CloudRig:
        cloud.consent.revoke()
        return cloud

    def test_ending_the_chat_revokes_consent_and_forgets_names(
        self, skill_ctx: SkillContext, chat: Chat, local_by_default: CloudRig
    ) -> None:
        service = AssistantService(lambda: skill_ctx)
        local_by_default.consent.grant()
        service.begin_chat()
        service.end_chat("closed")
        assert not local_by_default.consent.granted

    def test_hiding_revokes_consent(
        self, skill_ctx: SkillContext, local_by_default: CloudRig
    ) -> None:
        service = AssistantService(lambda: skill_ctx)
        assert service.ctx is skill_ctx  # the context is built lazily; consent lives in it
        local_by_default.consent.grant()
        service.revoke_consent()
        assert not local_by_default.consent.granted
        service.revoke_consent()  # harmless twice
