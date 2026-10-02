import tomllib

import pytest
from tests.core.conftest import Env
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed import cli
from vector_embed.core import runtime
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.providers.ollama import OllamaProvider


@pytest.fixture
def wired(env: Env, monkeypatch: pytest.MonkeyPatch) -> FakeOllamaClient:
    client = FakeOllamaClient(
        models={
            "qwen3-embedding:0.6b": {"caps": ["embedding"], "size": 600 * 1024**2},
            "llama3.2": {"caps": ["completion"], "size": 2 * 1024**3},
        }
    )
    provider = OllamaProvider(env.settings.embedding, client=client, sleep=lambda _s: None)
    monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
    monkeypatch.setattr(runtime, "build_provider", lambda _s: provider)
    monkeypatch.setattr(
        cli, "probe_hardware", lambda: Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
    )
    monkeypatch.setattr(
        "vector_embed.core.models.registry.probe_hardware",
        lambda: Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True),
    )
    return client


def settings_file(env: Env) -> dict[str, object]:
    with (env.data_dir / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_pull_streams_progress_to_stderr(
    wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "--pull", "qwen3.5:9b"]) == 0
    captured = capsys.readouterr()
    assert "pulled qwen3.5:9b" in captured.out
    assert "pulling 50%" in captured.err


def test_set_pins_and_clears_roles(
    env: Env, wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "--set", "chat=llama3.2"]) == 0
    assert "chat -> llama3.2" in capsys.readouterr().out
    assert settings_file(env)["models"] == {"overrides": {"chat": "llama3.2"}}
    assert cli.main(["models", "--set", "chat="]) == 0
    assert "chat -> automatic" in capsys.readouterr().out
    assert settings_file(env)["models"] == {"overrides": {}}


def test_unknown_role_is_a_usage_error(
    wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "--set", "poetry=x"]) == 2
    assert "unknown role" in capsys.readouterr().err


def test_embedder_switch_needs_yes(
    env: Env, wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "--embedder", "bge-m3"]) == 2
    assert "Confirm to continue" in capsys.readouterr().err
    assert not (env.data_dir / "settings.toml").exists()
    assert cli.main(["models", "--embedder", "bge-m3", "--yes"]) == 0
    assert "re-indexing" in capsys.readouterr().out
    assert settings_file(env)["embedding"] == {"model": "bge-m3"}
    assert env.state.get_meta("last_reconcile") == "0"


def test_selecting_the_current_embedder_is_a_noop(
    wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models", "--embedder", "qwen3-embedding:0.6b"]) == 0
    assert "already the embedder" in capsys.readouterr().out


def test_health_prints_the_dashboard(
    env: Env, wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    env.state.enqueue("x", delay=0)
    assert cli.main(["health"]) == 0
    out = capsys.readouterr().out
    assert "queue           : 1 total, 1 due" in out
    assert "cloud spend" in out
    assert "loaded models" in out
