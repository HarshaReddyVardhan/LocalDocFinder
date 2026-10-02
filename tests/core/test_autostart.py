import subprocess
from types import SimpleNamespace

import pytest

from vector_embed import cli
from vector_embed.core import autostart
from vector_embed.core.autostart import (
    TASKS,
    Autostart,
    AutostartError,
    AutostartTask,
    register_script,
    unregister_script,
)


def fake_command(entry: str) -> list[str]:
    return [r"C:\Program Files\VE\VectorEmbed.exe", entry]


def test_register_script_creates_both_tasks_with_the_battery_flags() -> None:
    script = register_script(TASKS, fake_command)
    assert "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries" in script
    assert "-ExecutionTimeLimit ([TimeSpan]::Zero)" in script
    assert "-MultipleInstances IgnoreNew" in script
    assert "-AtLogOn" in script
    assert "-RunLevel Limited" in script
    assert script.count("Register-ScheduledTask") == 2
    assert "-TaskName 'VectorEmbed Watcher'" in script
    assert "-TaskName 'VectorEmbed Search'" in script
    assert "-Execute 'C:\\Program Files\\VE\\VectorEmbed.exe' -Argument 'watcher'" in script
    assert "-Argument 'app'" in script


def test_arguments_with_spaces_and_quotes_are_escaped() -> None:
    def command(entry: str) -> list[str]:
        return ["C:\\it's here\\py.exe", "-m", "my pkg", entry]

    script = register_script((AutostartTask("T", "app"),), command)
    assert "-Execute 'C:\\it''s here\\py.exe'" in script  # PowerShell doubles a single quote
    assert "-Argument '-m \"my pkg\" app'" in script


def test_unregister_script_ignores_missing_tasks() -> None:
    script = unregister_script(TASKS)
    assert script.count("Unregister-ScheduledTask") == 2
    assert script.count("-ErrorAction SilentlyContinue") == 2
    assert "'VectorEmbed Watcher'" in script


def test_apply_registers_or_removes() -> None:
    scripts: list[str] = []

    def run(script: str) -> int:
        scripts.append(script)
        return 0

    tasks = Autostart(run, fake_command)
    tasks.apply(True)
    tasks.apply(False)
    assert "Register-ScheduledTask" in scripts[0]
    assert "Unregister-ScheduledTask" in scripts[1]


def test_failed_registration_raises() -> None:
    with pytest.raises(AutostartError, match="could not register"):
        Autostart(lambda _script: 1, fake_command).register()


def test_removing_never_raises() -> None:
    Autostart(lambda _script: 1, fake_command).unregister()


def test_default_command_runs_from_source_without_a_console() -> None:
    scripts: list[str] = []
    Autostart(lambda script: scripts.append(script) or 0).register()
    assert "-m vector_embed watcher" in scripts[0]


def test_run_powershell_reports_exit_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(returncode=3, stderr="denied")

    monkeypatch.setattr(autostart.subprocess, "run", fake_run)
    assert autostart.run_powershell("Get-Date") == 3
    assert calls[0][-1] == "Get-Date"


@pytest.mark.parametrize("error", [OSError("no powershell"), subprocess.TimeoutExpired("ps", 60)])
def test_run_powershell_survives_launch_failures(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise error

    monkeypatch.setattr(autostart.subprocess, "run", boom)
    assert autostart.run_powershell("x") == 1


def test_cli_autostart_on_and_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    applied: list[bool] = []
    monkeypatch.setattr(cli.Autostart, "apply", lambda _self, enabled: applied.append(enabled))
    assert cli.main(["autostart", "on"]) == 0
    assert cli.main(["autostart", "off"]) == 0
    assert applied == [True, False]
    out = capsys.readouterr().out
    assert "registered" in out
    assert "removed" in out


def test_cli_autostart_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_self: object, _enabled: bool) -> None:
        raise AutostartError("could not register the startup tasks")

    monkeypatch.setattr(cli.Autostart, "apply", fail)
    assert cli.main(["autostart", "on"]) == 1
    assert "could not register" in capsys.readouterr().err
