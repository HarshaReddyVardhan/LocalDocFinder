import tomllib

import pytest
from tests.core.conftest import Env
from tests.core.providers.fakes import FakeOllamaClient

from localdoc_finder import cli
from localdoc_finder.core import runtime
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.providers.ollama import OllamaProvider


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
        "localdoc_finder.core.models.registry.probe_hardware",
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


def test_health_shows_the_configured_monthly_budget(
    env: Env,
    wired: FakeOllamaClient,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localdoc_finder.core.settings import CloudSettings

    capped = env.settings.model_copy(update={"cloud": CloudSettings(monthly_budget_usd=5.0)})
    monkeypatch.setattr(cli, "load_settings", lambda: capped)
    env.state.record_usage("openrouter", "m", 10, 10, 1.25)
    assert cli.main(["health"]) == 0
    assert "$1.2500 of $5.00 this month" in capsys.readouterr().out
    env.state.record_usage("openrouter", "m", 10, 10, 4.0)  # now over it
    assert cli.main(["health"]) == 0
    assert "budget reached: cloud calls are paused" in capsys.readouterr().out


def test_health_without_a_budget_shows_only_the_spend(
    env: Env, wired: FakeOllamaClient, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["health"]) == 0
    out = capsys.readouterr().out
    assert "cloud spend     : $0.0000 this month" in out
    assert " of $" not in out
