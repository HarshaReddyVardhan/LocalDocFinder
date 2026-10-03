from pathlib import Path

import pytest
from tests.core.setup.fakes import Harness

from vector_embed.core.models.benchmark import (
    BenchKind,
    BenchmarkError,
    load_results,
)
from vector_embed.core.models.catalog import ROLE_EMBED
from vector_embed.core.providers.base import ProviderError
from vector_embed.core.setup.flow import (
    SETUP_COMPLETED_KEY,
    SetupCancelled,
    SetupError,
    SetupOptions,
    Stage,
)
from vector_embed.core.setup.ollama_install import OllamaState
from vector_embed.core.setup.plan import SetupChoices


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_full_run_downloads_benchmarks_and_saves(harness: Harness) -> None:
    result = harness.flow().run(SetupOptions())
    assert harness.host.pulled == ["qwen3-embedding:0.6b", "qwen3.5:9b"]
    assert harness.benched == ["qwen3-embedding:0.6b", "qwen3.5:9b"]  # embedder first, alone
    assert (result.embed_model, result.chat_model) == ("qwen3-embedding:0.6b", "qwen3.5:9b")
    assert result.downloaded == ("qwen3-embedding:0.6b", "qwen3.5:9b")
    assert harness.saved()["embedding"] == {"model": "qwen3-embedding:0.6b"}
    assert harness.saved()["models"] == {"overrides": {"chat": "qwen3.5:9b"}}
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) == "1234.0"
    assert {r.model for r in load_results(harness.state)} == {"qwen3-embedding:0.6b", "qwen3.5:9b"}
    stages = [e.stage for e in harness.events]
    assert stages[0] is Stage.OLLAMA
    assert stages[-1] is Stage.DONE
    assert any(e.stage is Stage.PULL and e.fraction == 0.5 for e in harness.events)


def test_rerun_skips_installed_models(tmp_path: Path) -> None:
    harness = Harness(tmp_path, installed=["qwen3-embedding:0.6b", "qwen3.5:9b"])
    result = harness.flow().run(SetupOptions())
    assert harness.host.pulled == []
    assert result.downloaded == ()


def test_interrupted_pull_is_resumed_by_running_again(harness: Harness) -> None:
    harness.host.fail_on = "qwen3.5:9b"
    with pytest.raises(ProviderError):
        harness.flow().run(SetupOptions())
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is None  # not marked done
    harness.host.fail_on = None
    harness.flow().run(SetupOptions())
    assert harness.host.pulled == [
        "qwen3-embedding:0.6b",
        "qwen3.5:9b",
    ]  # embedder kept, chat resumed
    assert harness.host.installed.count("qwen3-embedding:0.6b") == 1


def test_missing_ollama_needs_consent(tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    with pytest.raises(SetupError, match="not installed"):
        harness.flow().run(SetupOptions())
    assert harness.system.calls == []


def test_missing_ollama_is_installed_with_consent(tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    harness.flow().run(SetupOptions(install_ollama=True))
    assert any(call == "signature" for call in harness.system.calls)
    assert harness.system.up


def test_installed_but_stopped_ollama_is_started(tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    harness.system.exe = Path("ollama.exe")
    harness.flow().run(SetupOptions())
    assert harness.system.calls == ["spawn"]


def test_installer_failure_becomes_a_setup_error(tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    harness.system.installer_exit = 9
    with pytest.raises(SetupError, match="exit code 9"):
        harness.flow().run(SetupOptions(install_ollama=True))


def test_not_enough_disk_stops_before_any_download(harness: Harness) -> None:
    harness.system.free_mb = 3000
    with pytest.raises(SetupError, match="Not enough disk space"):
        harness.flow().run(SetupOptions())
    assert harness.host.pulled == []


def test_slow_model_offers_downgrade_and_accepts(harness: Harness) -> None:
    harness.model_rates = {"qwen3.5:9b": 2.0, "qwen3:8b": 25.0}
    harness.accept = True
    result = harness.flow().run(SetupOptions())
    assert [o.alternative for o in harness.offers] == ["qwen3:8b"]
    assert harness.offers[0].model == "qwen3.5:9b"
    assert result.chat_model == "qwen3:8b"
    assert "qwen3:8b" in harness.host.pulled
    assert harness.saved()["models"] == {"overrides": {"chat": "qwen3:8b"}}


def test_slow_model_kept_when_downgrade_declined(harness: Harness) -> None:
    harness.rates[BenchKind.CHAT] = 2.0
    result = harness.flow().run(SetupOptions())
    assert result.chat_model == "qwen3.5:9b"
    assert any("slow on this machine" in w for w in result.warnings)
    assert "qwen3:8b" not in harness.host.pulled


def test_failed_speed_test_is_a_warning_not_a_failure(harness: Harness) -> None:
    harness.bench_error = BenchmarkError("no output")
    result = harness.flow().run(SetupOptions())
    assert result.bench == ()
    assert any("could not speed-test" in w for w in result.warnings)
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is not None


def test_skip_bench(harness: Harness) -> None:
    result = harness.flow().run(SetupOptions(run_bench=False))
    assert harness.benched == []
    assert result.bench == ()


def test_existing_index_keeps_its_embedder(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness.state, "manifest_count", lambda: 40)
    result = harness.flow(current_embed="bge-m3").run(
        SetupOptions(choices=SetupChoices(embed="embeddinggemma"))
    )
    assert result.embed_model == "bge-m3"
    assert "bge-m3" in harness.host.pulled
    assert "bge-m3" not in harness.benched  # not re-measured, and nothing about it is changed
    assert any("index already exists" in w for w in result.warnings)


def test_user_choices_are_used(harness: Harness) -> None:
    result = harness.flow().run(
        SetupOptions(choices=SetupChoices(chat="llama3.2", embed="nomic-embed-text"))
    )
    assert (result.embed_model, result.chat_model) == ("nomic-embed-text", "llama3.2")


def test_extras_are_downloaded_but_not_benchmarked(harness: Harness) -> None:
    harness.flow().run(SetupOptions(choices=SetupChoices(extras=("caption",))))
    assert "qwen2.5vl:3b" in harness.host.pulled
    assert "qwen2.5vl:3b" not in harness.benched


def test_preview_changes_nothing(tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    preview = harness.flow().preview(SetupOptions())
    assert preview.ollama is OllamaState.MISSING
    assert preview.to_download == ("qwen3-embedding:0.6b", "qwen3.5:9b")
    assert preview.download_mb == 640 + 6100
    assert preview.enough_disk
    assert harness.system.calls == []
    assert harness.host.pulled == []
    assert not harness.settings_path.exists()


def test_preview_lists_what_is_still_missing(harness: Harness) -> None:
    harness.host.installed = ["qwen3.5:9b"]
    preview = harness.flow().preview(SetupOptions())
    assert preview.ollama is OllamaState.RUNNING
    assert preview.to_download == ("qwen3-embedding:0.6b",)
    assert preview.download_mb == 640
    assert ROLE_EMBED in {m.role for m in preview.plan.models}


class TestDiskBeforeInstallingOllama:
    def test_too_little_room_stops_before_downloading_the_installer(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, up=False)
        harness.system.free_mb = 2000  # the installer plus the program need far more
        with pytest.raises(SetupError, match="Not enough disk space to install Ollama"):
            harness.flow().run(SetupOptions(install_ollama=True))
        assert harness.system.calls == []  # nothing was downloaded
        assert tmp_path / "data" / "downloads" in harness.system.queried

    def test_enough_room_installs_as_before(self, tmp_path: Path) -> None:
        harness = Harness(tmp_path, up=False)
        harness.system.free_mb = 50_000
        harness.flow().run(SetupOptions(install_ollama=True))
        assert harness.system.up


class TestCancellation:
    def test_cancelling_before_the_run_does_nothing(self, harness: Harness) -> None:
        flow = harness.flow()
        flow.cancelled = lambda: True
        with pytest.raises(SetupCancelled):
            flow.run(SetupOptions())
        assert harness.host.pulled == []
        assert harness.state.get_meta(SETUP_COMPLETED_KEY) is None

    def test_cancelling_between_model_downloads_stops_the_next_one(self, harness: Harness) -> None:
        stop = False

        def progress(event: object) -> None:
            nonlocal stop
            if harness.host.pulled:  # the first model finished: the user closes the wizard
                stop = True

        flow = harness.flow(progress=progress)
        flow.cancelled = lambda: stop
        with pytest.raises(SetupCancelled):
            flow.run(SetupOptions())
        assert harness.host.pulled == ["qwen3-embedding:0.6b"]  # not the second
        assert harness.state.get_meta(SETUP_COMPLETED_KEY) is None  # and it is not "complete"

    def test_cancelling_during_the_installer_download_abandons_the_install(
        self, tmp_path: Path
    ) -> None:
        harness = Harness(tmp_path, up=False)
        flow = harness.flow()
        flow.cancelled = lambda: any(c.startswith("download") for c in harness.system.calls)
        with pytest.raises(SetupCancelled):
            flow.run(SetupOptions(install_ollama=True))
        assert "signature" not in harness.system.calls  # never verified, never run
        assert not any(call.startswith("run") for call in harness.system.calls)
        assert not (tmp_path / "data" / "downloads" / "OllamaSetup.exe").exists()  # cleaned up

    def test_a_cancelled_run_can_be_started_again(self, harness: Harness) -> None:
        flow = harness.flow()
        flow.cancelled = lambda: True
        with pytest.raises(SetupCancelled):
            flow.run(SetupOptions())
        flow.cancelled = lambda: False
        assert flow.run(SetupOptions()).chat_model == "qwen3.5:9b"
