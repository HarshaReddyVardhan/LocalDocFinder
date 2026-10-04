import asyncio
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError
from tests.core.conftest import Chat, Env, Power
from tests.core.match.test_pipeline import JD, faithful_model, resume

from localdoc_finder import mcp_server
from localdoc_finder.core.llm import CHAT_LOCK
from localdoc_finder.core.privacy.policy import PrivacyFilter
from localdoc_finder.core.rag import NOT_FOUND
from localdoc_finder.core.settings import PrivacySettings
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.mcp_server import Service, build_server

PAYMENTS = (
    "def retry_failed_payments(order):\n"
    "    # retry the charge with exponential backoff\n"
    "    return charge_card(order)\n"
)
LEAKY = "# Notes\n\nRetry failed payments. Passport No: K1234567 and SSN: 123-45-6789 on file.\n"
PRIVATE = "# Private\n\nRetry failed payments using the secret exponential backoff plan.\n"


def write(env: Env, rel: str, text: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def ctx(env: Env, chat: Chat, skill_ctx: SkillContext) -> SkillContext:
    privacy_settings = PrivacySettings(never_send_globs=("**/private/**",))
    skill_ctx.settings = skill_ctx.settings.model_copy(update={"privacy": privacy_settings})
    skill_ctx.extras["privacy"] = PrivacyFilter(privacy_settings, env.scope)
    return skill_ctx


@pytest.fixture
def indexed(env: Env, ctx: SkillContext) -> dict[str, str]:
    paths = {
        "code": write(env, "payments.py", PAYMENTS),
        "leaky": write(env, "notes.md", LEAKY),
        "private": write(env, "private/plan.md", PRIVATE),
    }
    env.indexer.index_paths(list(paths.values()))
    env.store.maintain()
    return paths


@pytest.fixture
def service(ctx: SkillContext) -> Service:
    return Service(ctx)


class TestSearch:
    def test_returns_locations_and_snippets(
        self, service: Service, indexed: dict[str, str]
    ) -> None:
        response = service.search("retry failed payments", 5)
        hit = next(h for h in response.results if h.path == indexed["code"])
        assert hit.start_line >= 1
        assert "retry" in hit.snippet.lower()

    def test_private_files_never_appear(self, service: Service, indexed: dict[str, str]) -> None:
        response = service.search("secret exponential backoff plan", 10)
        assert indexed["private"] not in {h.path for h in response.results}
        assert response.withheld >= 1

    def test_ids_are_masked_in_snippets(self, service: Service, indexed: dict[str, str]) -> None:
        response = service.search("passport SSN on file", 10)
        snippet = next(h.snippet for h in response.results if h.path == indexed["leaky"])
        assert "K1234567" not in snippet
        assert "123-45-6789" not in snippet
        assert "REMOVED" in snippet

    def test_removing_private_files_still_fills_the_limit(
        self, service: Service, indexed: dict[str, str]
    ) -> None:
        assert len(service.search("retry failed payments", 2).results) == 2

    def test_battery_policy_is_reported(
        self, service: Service, power_state: Power, ctx: SkillContext
    ) -> None:
        ctx.power._settings = ctx.settings.power.model_copy(update={"search_on_battery": False})
        power_state.on_ac = False
        ctx.power.update()
        with pytest.raises(RuntimeError, match="battery"):
            service.search("anything", 3)


class TestAsk:
    def test_answers_with_the_cited_sources_only(
        self, service: Service, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["Retry with backoff [1]."]
        response = service.ask("how do we retry failed payments", None)
        assert response.answer == "Retry with backoff [1]."
        assert [s.n for s in response.sources] == [1]
        assert response.sources[0].path != indexed["private"]
        assert not response.not_found

    def test_private_files_are_kept_out_of_the_prompt(
        self, service: Service, chat: Chat, indexed: dict[str, str]
    ) -> None:
        service.ask("retry failed payments backoff", None)
        prompts = " ".join(str(c["messages"]) for c in chat.chat_calls())
        assert "retry the charge" in prompts  # the public file is there
        assert "using the secret" not in prompts  # the private one is not

    def test_answer_text_is_masked(
        self, service: Service, chat: Chat, indexed: dict[str, str]
    ) -> None:
        chat.client.chat_reply = ["The passport is Passport No: K1234567 [1]."]
        response = service.ask("passport on file", None)
        assert "K1234567" not in response.answer

    def test_nothing_indexed_is_not_found_without_the_model(
        self, service: Service, chat: Chat
    ) -> None:
        response = service.ask("anything", None)
        assert response.answer == NOT_FOUND
        assert response.not_found
        assert chat.chat_calls() == []

    def test_models_are_released_and_the_lock_is_free_afterwards(
        self, service: Service, chat: Chat, env: Env, indexed: dict[str, str]
    ) -> None:
        service.ask("how do we retry failed payments", None)
        assert not chat.gateway.session_active
        assert env.state.acquire_lock(CHAT_LOCK, "someone-else", 60)

    def test_a_chat_session_held_by_the_app_blocks_it(
        self, service: Service, env: Env, indexed: dict[str, str]
    ) -> None:
        assert env.state.acquire_lock(CHAT_LOCK, "the-app-window", 600)
        with pytest.raises(RuntimeError, match="another chat session"):
            service.ask("how do we retry failed payments", None)

    def test_the_lock_is_released_when_the_model_fails(
        self, service: Service, chat: Chat, env: Env, indexed: dict[str, str], power_state: Power
    ) -> None:
        power_state.on_ac = False
        chat.gateway._power.update()
        with pytest.raises(RuntimeError, match="battery"):
            service.ask("how do we retry failed payments", None)
        assert env.state.acquire_lock(CHAT_LOCK, "someone-else", 60)


class TestMatch:
    @pytest.fixture
    def resumes(self, env: Env, chat: Chat, ctx: SkillContext) -> dict[str, str]:
        chat.client.chat_json_fn = faithful_model
        paths = {
            "a": write(env, "Resume_a.txt", resume("Jane", "Python", "PostgreSQL", "AWS")),
            "b": write(env, "Resume_b.txt", resume("Sam", "Python")),
            "private": write(env, "private/Resume_p.txt", resume("Pat", "Python", "PostgreSQL")),
        }
        env.indexer.index_paths(list(paths.values()))
        env.store.maintain()
        return paths

    def test_ranks_documents_and_returns_scores_only(
        self, service: Service, resumes: dict[str, str]
    ) -> None:
        response = service.match(JD, "resume", 5)
        assert response.ranked[0].path == resumes["a"]
        assert response.ranked[0].score == 83  # 2/2 must-haves, 1/2 nice-to-haves
        assert "must-haves met" in response.ranked[0].summary
        assert [e.rank for e in response.ranked] == list(range(1, len(response.ranked) + 1))

    def test_private_documents_are_not_scored(
        self, service: Service, resumes: dict[str, str]
    ) -> None:
        response = service.match(JD, "resume", 5)
        assert resumes["private"] not in {e.path for e in response.ranked}
        assert response.withheld == 1

    def test_models_are_released_afterwards(
        self, service: Service, chat: Chat, resumes: dict[str, str]
    ) -> None:
        service.match(JD, "resume", 2)
        assert not chat.gateway.session_active


class TestNoCloud:
    def test_the_context_has_no_cloud_route(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from localdoc_finder.core import runtime
        from localdoc_finder.core.secrets import MemoryKeyStore
        from localdoc_finder.core.settings import (
            CloudProviderSettings,
            CloudSettings,
            Settings,
            StorageSettings,
        )

        settings = Settings(
            storage=StorageSettings(data_dir=env.data_dir),
            cloud=CloudSettings(
                providers={
                    "openrouter": CloudProviderSettings(
                        base_url="https://openrouter.ai/api/v1", models={"chat": "vendor/chat"}
                    )
                },
                active="openrouter",
                routing={"chat": "cloud"},
            ),
        )
        keys = MemoryKeyStore({"openrouter": "sk-test-1234567890abcdef"})
        ctx = runtime.build_skill_context(
            settings, env.state, keys=keys, owner="mcp", allow_cloud=False
        )
        assert ctx.extras["llm"].router is None
        assert not ctx.extras["llm"].will_use_cloud()


def call(server: Any, name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(server.call_tool(name, arguments))


class TestServer:
    @pytest.fixture
    def server(self, service: Service) -> Any:
        return build_server(lambda: service)

    def test_exposes_three_read_only_tools(self, server: Any) -> None:
        tools = asyncio.run(server.list_tools())
        assert {t.name for t in tools} == {"search", "ask", "match"}
        assert all(t.annotations and t.annotations.read_only_hint for t in tools)

    def test_tools_for_features_that_are_off_are_not_offered(self, service: Service) -> None:
        tools = asyncio.run(build_server(lambda: service, frozenset({"match"})).list_tools())
        assert {t.name for t in tools} == {"search", "match"}
        tools = asyncio.run(build_server(lambda: service, frozenset()).list_tools())
        assert {t.name for t in tools} == {"search"}

    def test_search_over_the_protocol_returns_structured_content(
        self, server: Any, indexed: dict[str, str]
    ) -> None:
        result = call(server, "search", {"query": "retry failed payments", "limit": 3})
        assert not result.is_error
        paths = [h["path"] for h in result.structured_content["results"]]
        assert indexed["code"] in paths

    def test_limit_is_capped(self, server: Any) -> None:
        with pytest.raises(ToolError):
            call(server, "search", {"query": "x", "limit": mcp_server.MAX_RESULTS + 1})

    def test_policy_errors_become_tool_errors(
        self, server: Any, env: Env, indexed: dict[str, str]
    ) -> None:
        env.state.acquire_lock(CHAT_LOCK, "the-app-window", 600)
        with pytest.raises(ToolError, match="another chat session"):
            call(server, "ask", {"question": "how do we retry failed payments"})

    def test_nothing_is_built_until_the_first_call(self) -> None:
        built: list[int] = []

        def factory() -> Service:
            built.append(1)
            raise AssertionError("not reached")

        server = build_server(factory)
        asyncio.run(server.list_tools())
        assert built == []
