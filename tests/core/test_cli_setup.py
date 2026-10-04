import argparse
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.core.conftest import Env
from tests.core.setup.fakes import Harness

from localdoc_finder import cli, cli_setup
from localdoc_finder.core.features import FEATURES
from localdoc_finder.core.models.benchmark import BenchKind
from localdoc_finder.core.setup.flow import (
    SETUP_COMPLETED_KEY,
    SetupError,
    SetupEvent,
    SetupFlow,
    SlowOffer,
    Stage,
)


def args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "yes": False,
        "embed": None,
        "chat": None,
        "extras": [],
        "features": list(FEATURES),  # None would keep the features already on
        "dry_run": False,
        "no_install_ollama": False,
        "skip_bench": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class Console:
    def __init__(self) -> None:
        self.out: list[str] = []
        self.err: list[str] = []
        self.questions: list[str] = []
        self.answer = True

    def ask(self, question: str) -> bool:
        self.questions.append(question)
        return self.answer


def run(harness: Harness, console: Console, env: Env, **overrides: object) -> int:
    def build(
        settings: object,
        state: object,
        progress: Callable[[SetupEvent], None],
        accept: Callable[[SlowOffer], bool],
    ) -> SetupFlow:
        return harness.flow(progress=progress, accept=accept)

    return cli_setup.run_setup(
        args(**overrides),
        env.settings,
        console.out.append,
        console.err.append,
        ask=console.ask,
        build=build,  # type: ignore[arg-type]  # the fake builder ignores settings and state
    )


@pytest.fixture
def harness(env: Env) -> Harness:
    return Harness(env.data_dir.parent / "cli")


def test_without_features_the_ones_already_on_are_kept(harness: Harness, env: Env) -> None:
    console = Console()
    assert run(harness, console, env, dry_run=True, features=None) == 0
    text = "\n".join(console.out)
    assert "embed: qwen3-embedding:0.6b (640 MB)" in text
    assert "chat:" not in text  # a fresh install is search only


def test_dry_run_prints_the_plan_and_changes_nothing(harness: Harness, env: Env) -> None:
    console = Console()
    assert run(harness, console, env, dry_run=True) == 0
    text = "\n".join(console.out)
    assert "embed: qwen3-embedding:0.6b (640 MB)" in text
    assert "chat: qwen3.5:9b (6100 MB)" in text
    assert "To download: 6740 MB" in text
    assert harness.host.pulled == []
    assert console.questions == []


def test_yes_runs_everything_without_asking(harness: Harness, env: Env) -> None:
    console = Console()
    assert run(harness, console, env, yes=True) == 0
    assert console.questions == []
    assert harness.host.pulled == ["qwen3-embedding:0.6b", "qwen3.5:9b"]
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is not None
    assert "Chat model: qwen3.5:9b" in console.out[-1]
    assert any("[pull]" in line for line in console.err)


def test_declining_the_download_cancels(harness: Harness, env: Env) -> None:
    console = Console()
    console.answer = False
    assert run(harness, console, env) == 1
    assert console.questions == ["Download 6740 MB and continue?"]
    assert harness.host.pulled == []
    assert "Cancelled." in console.err


def test_missing_ollama_asks_before_installing(env: Env) -> None:
    harness = Harness(env.data_dir.parent / "cli2", up=False)
    console = Console()
    assert run(harness, console, env) == 0
    assert any("Ollama is not installed" in q for q in console.questions)
    assert harness.system.up


def test_no_install_ollama_is_an_unconditional_no(env: Env) -> None:
    harness = Harness(env.data_dir.parent / "cli3", up=False)
    console = Console()
    with pytest.raises(SetupError, match="not installed"):
        run(harness, console, env, yes=True, no_install_ollama=True)
    assert harness.system.calls == []


def test_not_enough_disk_is_reported(harness: Harness, env: Env) -> None:
    harness.system.free_mb = 100
    console = Console()
    assert run(harness, console, env, yes=True) == 1
    assert "Not enough disk space for the models." in console.err
    assert harness.host.pulled == []


def test_slow_model_prompts_for_downgrade(harness: Harness, env: Env) -> None:
    harness.model_rates = {"qwen3.5:9b": 2.0, "qwen3:8b": 25.0}
    console = Console()
    assert run(harness, console, env) == 0
    assert any("Switch to the smaller qwen3:8b?" in q for q in console.questions)
    assert "Chat model: qwen3:8b" in console.out[-1]
    assert harness.rates[BenchKind.CHAT] == 30.0


def test_skip_bench(harness: Harness, env: Env) -> None:
    console = Console()
    assert run(harness, console, env, yes=True, skip_bench=True) == 0
    assert harness.benched == []


def test_progress_printer_throttles_downloads() -> None:
    lines: list[str] = []
    printer = cli_setup._ProgressPrinter(lines.append)
    for fraction in (0.01, 0.02, 0.05, 0.11, 0.12, 0.5):
        printer(SetupEvent(Stage.PULL, "m: pulling", fraction))
    printer(SetupEvent(Stage.DISK, "100 MB free"))
    assert lines == [
        "[pull] m: pulling 1%",
        "[pull] m: pulling 11%",
        "[pull] m: pulling 50%",
        "[disk] 100 MB free",
    ]


def test_main_dispatches_ve_setup(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
    monkeypatch.setattr(cli, "run_setup", lambda a, s, o, e: seen.append(a) or 0)
    assert cli.main(["setup", "--dry-run", "--extras", "caption"]) == 0
    assert seen[0].dry_run
    assert seen[0].extras == ["caption"]


def test_main_rejects_unknown_extras(
    env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "load_settings", lambda: env.settings)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["setup", "--extras", "bogus"])
    assert exit_info.value.code == 2
    assert "bogus" in capsys.readouterr().err
