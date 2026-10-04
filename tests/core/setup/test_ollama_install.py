from pathlib import Path

import pytest
from tests.core.setup.fakes import Clock, FakeSystem

from localdoc_finder.core.setup.ollama_install import (
    EXPECTED_SIGNER,
    INSTALLER_ARGS,
    INSTALLER_URL,
    OllamaSetup,
    OllamaSetupError,
    OllamaState,
    Signature,
)


def make(system: FakeSystem, wait: float = 10.0) -> OllamaSetup:
    clock = Clock()
    return OllamaSetup(system, sleep=clock.sleep, clock=clock, server_wait_seconds=wait)


def no_progress(done: int, total: int | None) -> None:
    return None


def test_detect_running_installed_and_missing() -> None:
    system = FakeSystem()
    setup = make(system)
    assert setup.detect() is OllamaState.MISSING
    system.exe = Path("ollama.exe")
    assert setup.detect() is OllamaState.INSTALLED_NOT_RUNNING
    system.up = True
    assert setup.detect() is OllamaState.RUNNING


def test_install_downloads_verifies_runs_and_cleans_up(tmp_path: Path) -> None:
    system = FakeSystem()
    seen: list[tuple[int, int | None]] = []
    make(system).install(tmp_path, lambda done, total: seen.append((done, total)), consented=True)
    assert system.calls == [f"download {INSTALLER_URL}", "signature", f"run {INSTALLER_ARGS}"]
    assert seen == [(2, 2)]
    assert not (tmp_path / "OllamaSetup.exe").exists()


def test_install_refuses_without_consent(tmp_path: Path) -> None:
    system = FakeSystem()
    with pytest.raises(OllamaSetupError, match="consent"):
        make(system).install(tmp_path, no_progress, consented=False)
    assert system.calls == []


@pytest.mark.parametrize(
    "signature",
    [Signature(valid=False, signer=EXPECTED_SIGNER), Signature(valid=True, signer="Mallory Ltd")],
    ids=["invalid", "wrong-signer"],
)
def test_install_never_runs_an_unverified_installer(tmp_path: Path, signature: Signature) -> None:
    system = FakeSystem()
    system.sig = signature
    with pytest.raises(OllamaSetupError, match="not running it"):
        make(system).install(tmp_path, no_progress, consented=True)
    assert not any(call.startswith("run") for call in system.calls)
    assert not (tmp_path / "OllamaSetup.exe").exists()


def test_installer_failure_is_reported(tmp_path: Path) -> None:
    system = FakeSystem()
    system.installer_exit = 5
    with pytest.raises(OllamaSetupError, match="exit code 5"):
        make(system).install(tmp_path, no_progress, consented=True)


def test_run_installer_waits_for_the_server(tmp_path: Path) -> None:
    system = FakeSystem()
    system.up_after_pings = 3
    make(system).install(tmp_path, no_progress, consented=True)
    assert system.up


def test_server_that_never_starts_times_out() -> None:
    system = FakeSystem()
    system.exe = Path("ollama.exe")
    system.up_after_pings = 10_000
    with pytest.raises(OllamaSetupError, match="did not start"):
        make(system, wait=3.0).start_server()


def test_start_server_spawns_only_when_needed() -> None:
    system = FakeSystem()
    system.exe = Path("ollama.exe")
    make(system).start_server()
    assert system.calls == ["spawn"]
    make(system).start_server()  # already up
    assert system.calls == ["spawn"]


def test_start_server_without_install_fails() -> None:
    with pytest.raises(OllamaSetupError, match="not installed"):
        make(FakeSystem()).start_server()


def test_free_disk_checks_the_model_store_drive() -> None:
    system = FakeSystem()
    system.free_mb = 1234
    assert make(system).free_disk_mb() == 1234
    assert system.queried == [Path("models")]


def test_download_installer_creates_the_directory(tmp_path: Path) -> None:
    target = make(FakeSystem()).download_installer(tmp_path / "a" / "b", no_progress)
    assert target.read_bytes() == b"MZ"
