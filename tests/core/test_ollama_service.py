from pathlib import Path

import pytest
from tests.core.setup.fakes import Clock, FakeSystem

from localdoc_finder.core.ollama_service import AUTOSTART_DISABLE_ENV, ensure_ollama_running
from localdoc_finder.core.setup.ollama_install import OllamaSetup, OllamaState


@pytest.fixture(autouse=True)
def real_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(AUTOSTART_DISABLE_ENV)  # these tests inject a fake machine instead


def setup_for(system: FakeSystem) -> OllamaSetup:
    clock = Clock()
    return OllamaSetup(system, sleep=clock.sleep, clock=clock)


def test_a_running_server_is_used_as_it_is() -> None:
    system = FakeSystem()
    system.up = True
    assert ensure_ollama_running("host", setup=setup_for(system)) is OllamaState.RUNNING
    assert "spawn" not in system.calls


def test_an_installed_but_stopped_server_is_started() -> None:
    system = FakeSystem()
    system.exe = Path("ollama.exe")
    assert ensure_ollama_running("host", setup=setup_for(system)) is OllamaState.RUNNING
    assert system.calls == ["spawn"]


def test_a_missing_ollama_is_reported_and_nothing_is_installed() -> None:
    system = FakeSystem()
    assert ensure_ollama_running("host", setup=setup_for(system)) is OllamaState.MISSING
    assert system.calls == []


def test_a_server_that_will_not_come_up_is_reported_not_raised() -> None:
    system = FakeSystem()
    system.exe = Path("ollama.exe")
    system.up_after_pings = 10**6  # never answers
    state = ensure_ollama_running("host", setup=setup_for(system))
    assert state is OllamaState.INSTALLED_NOT_RUNNING


def test_the_env_switch_disables_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AUTOSTART_DISABLE_ENV, "1")
    system = FakeSystem()
    assert ensure_ollama_running("host", setup=setup_for(system)) is OllamaState.RUNNING
    assert system.pings == 0
