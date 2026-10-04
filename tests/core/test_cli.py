import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import Field
from tests.core.conftest import Chat, Env
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed import cli
from vector_embed.core import runtime
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.providers.base import ModelInfo, ProviderError
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings import FeatureSettings, SettingsError
from vector_embed.core.setup.ollama_install import OllamaState
from vector_embed.core.skills.base import (
    SKILLS,
    Skill,
    SkillContext,
    SkillInput,
    register_skill,
)


@pytest.fixture
def wired(env: Env, skill_ctx: SkillContext, monkeypatch: pytest.MonkeyPatch) -> SkillContext:
    """Make the CLI use the test environment instead of the real machine."""
    monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
    monkeypatch.setattr(runtime, "build_skill_context", lambda _s, _st: skill_ctx)
    return skill_ctx


def index_file(env: Env, name: str, text: str) -> None:
    path = env.root / name
    path.write_text(text, encoding="utf-8")
    env.indexer.index_paths([str(path)])
    env.store.maintain()


class TestSearchCommand:
    def test_search_prints_ranked_results(
        self, env: Env, wired: SkillContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        index_file(env, "payments.py", "def retry_failed_payments():\n    return 1\n")
        assert cli.main(["search", "retry", "failed", "payments", "--limit", "3"]) == 0
        assert "payments.py" in capsys.readouterr().out

    def test_no_results(self, wired: SkillContext, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.main(["search", "nothing"]) == 0
        assert capsys.readouterr().out.strip() == "no results"

    def test_data_dir_override_is_applied(
        self, env: Env, wired: SkillContext, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["--data-dir", str(tmp_path / "elsewhere"), "search", "x"]) == 0

    def test_battery_policy_errors_exit_1(
        self, env: Env, wired: SkillContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        wired.power._settings = env.settings.power.model_copy(update={"search_on_battery": False})
        wired.power._probe = lambda: False
        assert cli.main(["search", "x"]) == 1
        assert "disabled on battery" in capsys.readouterr().err


class TestErrors:
    def test_bad_settings_exit_2(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def broken() -> None:
            raise SettingsError("bad toml")

        monkeypatch.setattr(cli, "load_settings", broken)
        assert cli.main(["status"]) == 2
        assert "settings error: bad toml" in capsys.readouterr().err

    def test_provider_errors_exit_1(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def down(_s: object, _st: object) -> None:
            raise ProviderError("ollama down")

        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        monkeypatch.setattr(runtime, "build_skill_context", down)
        assert cli.main(["search", "x"]) == 1
        assert "ollama down" in capsys.readouterr().err

    def test_a_command_is_required(self) -> None:
        with pytest.raises(SystemExit) as exit_info:
            cli.main([])
        assert exit_info.value.code == 2


class TestOtherCommands:
    def test_status(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        assert cli.main(["status"]) == 0
        assert "indexed files" in capsys.readouterr().out

    def test_docs_lists_the_indexed_files(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        env.state.manifest_set("D:/Work/Plan.md", 1, 4096, "h")
        env.state.manifest_set("D:/Work/Other.txt", 1, 10, "h2")
        assert cli.main(["docs", "plan"]) == 0
        out = capsys.readouterr().out
        assert "Plan.md" in out
        assert "Other.txt" not in out
        assert "indexed in total: 2" in out

    def test_model_commands_start_ollama_first(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        asked: list[str] = []
        monkeypatch.setattr(
            cli, "ensure_ollama_running", lambda host: asked.append(host) or OllamaState.MISSING
        )
        monkeypatch.setattr(cli, "cmd_health", lambda _s: 0)
        assert cli.main(["health"]) == 0
        assert asked == [env.settings.ollama_host]
        assert "Ollama is not running" in capsys.readouterr().err
        assert cli.main(["status"]) == 0  # no model needed: no check
        assert len(asked) == 1

    def test_index_delegates_to_the_worker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[list[str]] = []
        monkeypatch.setattr(cli, "load_settings", lambda: None)
        monkeypatch.setattr(cli.worker, "main", lambda argv: seen.append(list(argv)) or 0)
        assert cli.main(["index", "--now", "--path", "D:/x"]) == 0
        assert seen == [["--now", "--path", "D:/x"]]

    def test_models_report(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        client = FakeOllamaClient(
            models={
                "qwen3-embedding:0.6b": {"caps": ["embedding"], "size": 600 * 1024**2},
                "mxbai-embed-large": {"caps": ["embedding"], "size": 700 * 1024**2},
                "deepseek-r1:8b": {"caps": ["completion"], "size": 5 * 1024**3},
                "llama3.2": {"caps": ["completion"], "size": 2 * 1024**3},
            }
        )
        provider = OllamaProvider(env.settings.embedding, client=client)
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        monkeypatch.setattr(runtime, "build_provider", lambda _s: provider)
        monkeypatch.setattr(
            cli, "probe_hardware", lambda: Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
        )
        assert cli.main(["models"]) == 0
        text = capsys.readouterr().out
        assert "mxbai-embed-large" in text
        assert "512-token limit" in text
        assert "deepseek-r1:8b" in text
        assert "unused" in text
        assert "ollama pull qwen3.5:9b" in text

    def test_doctor_reports_failures(
        self, env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        provider = OllamaProvider(env.settings.embedding, client=FakeOllamaClient())
        monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
        monkeypatch.setattr(runtime, "build_provider", lambda _s: provider)
        monkeypatch.setattr(
            cli, "probe_hardware", lambda: Hardware(None, 0, 0, 16000, 8000, 4, False)
        )
        assert cli.main(["doctor"]) == 1
        text = capsys.readouterr().out
        assert "[FAIL] ollama" in text
        assert "[FAIL] embedding model" in text
        assert "ollama pull qwen3-embedding:0.6b" in text
        assert "battery" in text


class TestGeneratedArguments:
    def make_parser(
        self, input_cls: type[SkillInput], positional: str | None
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        cli.add_input_arguments(parser, dict(input_cls.model_fields), positional)
        return parser

    def test_all_supported_field_types(self) -> None:
        class Demo(SkillInput):
            text: str = Field(description="free text")
            count: int | None = None
            ratio: float = 0.5
            flag: bool = False
            names: list[str] = Field(default_factory=list)
            must: str

        parser = self.make_parser(Demo, "text")
        args = parser.parse_args(
            ["hello", "world", "--must", "x", "--count", "3", "--ratio", "0.25", "--flag",
             "--names", "a", "b"]
        )  # fmt: skip
        values = cli.collect_input(args, dict(Demo.model_fields), "text")
        assert values == {
            "text": "hello world", "must": "x", "count": 3, "ratio": 0.25, "flag": True,
            "names": ["a", "b"],
        }  # fmt: skip
        assert Demo(**values).count == 3

    def test_negated_bool_and_defaults_are_omitted(self) -> None:
        class Demo(SkillInput):
            flag: bool = True

        args = self.make_parser(Demo, None).parse_args(["--no-flag"])
        assert cli.collect_input(args, dict(Demo.model_fields), None) == {"flag": False}
        args = self.make_parser(Demo, None).parse_args([])
        assert cli.collect_input(args, dict(Demo.model_fields), None) == {}

    def test_unsupported_type_is_rejected(self) -> None:
        class Demo(SkillInput):
            blob: dict[str, int] = Field(default_factory=dict)

        with pytest.raises(ValueError, match="unsupported"):
            self.make_parser(Demo, None)


class TestNewSkillAppearsInTheCli:
    def test_a_registered_skill_gets_a_command_and_streams(
        self, wired: SkillContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class EchoInput(SkillInput):
            words: str

        @register_skill("echo")
        class Echo(Skill):
            name = "echo"
            title = "Echo"
            description = "echo words back"
            Input = EchoInput
            cli_positional = "words"

            def run(self, params: SkillInput) -> object:
                return "unused"

            def stream(self, params: SkillInput) -> Iterator[str]:
                assert isinstance(params, EchoInput)
                yield from (w + " " for w in params.words.split())

        try:
            assert cli.main(["echo", "a", "b"]) == 0
            assert capsys.readouterr().out == "a b \n"
        finally:
            SKILLS.remove("echo")


def test_model_info_is_exposed_to_the_report() -> None:
    info = ModelInfo("m", "ollama")
    assert not info.is_embedding


def with_features(env: Env, monkeypatch: pytest.MonkeyPatch, **on: bool) -> None:
    settings = env.settings.model_copy(update={"features": FeatureSettings(**on)})
    monkeypatch.setattr(cli, "load_settings", lambda: settings)


class TestOptionalFeatures:
    def test_a_feature_that_is_off_is_refused_with_a_way_to_turn_it_on(
        self, wired: SkillContext, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["chat", "hello"]) == cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert "ve chat is not enabled" in err
        assert "ve setup --features chat" in err


class TestChatSessions:
    @pytest.fixture(autouse=True)
    def chat_on(self, env: Env, wired: SkillContext, monkeypatch: pytest.MonkeyPatch) -> None:
        with_features(env, monkeypatch, chat=True)

    def test_list_sessions_needs_no_message(
        self, env: Env, wired: SkillContext, chat: Chat, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["chat", "--list-sessions"]) == 0
        assert capsys.readouterr().out.strip() == "no chat sessions yet"
        chat.client.chat_reply = ["Hi."]
        assert cli.main(["chat", "hello", "there"]) == 0
        capsys.readouterr()
        assert cli.main(["chat", "--list-sessions"]) == 0
        listed = capsys.readouterr().out
        assert "hello there" in listed and "2 msgs" in listed

    def test_a_chat_without_a_message_is_a_usage_error(
        self, wired: SkillContext, chat: Chat, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.main(["chat"]) == cli.EXIT_USAGE
        assert "ve chat: a message is required" in capsys.readouterr().err
