import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from vector_embed.core.setup import windows_ollama
from vector_embed.core.setup.ollama_install import Signature
from vector_embed.core.setup.windows_ollama import WindowsOllamaSystem


def system_with(transport: httpx.MockTransport | None = None) -> WindowsOllamaSystem:
    client = httpx.Client(transport=transport, follow_redirects=True) if transport else None
    return WindowsOllamaSystem(client=client)


def test_ping_true_on_200_false_on_error() -> None:
    ok = system_with(httpx.MockTransport(lambda r: httpx.Response(200, json={"version": "1"})))
    assert ok.ping()
    down = system_with(httpx.MockTransport(lambda r: httpx.Response(500)))
    assert not down.ping()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert not system_with(httpx.MockTransport(refuse)).ping()


def test_download_writes_file_and_reports_progress(tmp_path: Path) -> None:
    body = b"x" * 1000
    system = system_with(httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    seen: list[tuple[int, int | None]] = []
    target = tmp_path / "OllamaSetup.exe"
    system.download("https://example.test/x", target, lambda d, t: seen.append((d, t)))
    assert target.read_bytes() == body
    assert seen[-1][0] == 1000
    assert [p.name for p in tmp_path.iterdir()] == ["OllamaSetup.exe"]


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    system = system_with(httpx.MockTransport(lambda r: httpx.Response(404)))
    target = tmp_path / "OllamaSetup.exe"
    with pytest.raises(httpx.HTTPStatusError):
        system.download("https://example.test/x", target, lambda d, t: None)
    assert list(tmp_path.iterdir()) == []


def test_find_executable_prefers_path_then_local_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(windows_ollama.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    system = system_with()
    assert system.find_executable() is None
    exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert system.find_executable() == exe
    monkeypatch.setattr(windows_ollama.shutil, "which", lambda name: "C:/bin/ollama.exe")
    assert system.find_executable() == Path("C:/bin/ollama.exe")


def test_models_dir_honours_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    assert system_with().models_dir() == tmp_path
    monkeypatch.delenv("OLLAMA_MODELS")
    assert system_with().models_dir() == Path.home() / ".ollama" / "models"


def test_free_disk_uses_nearest_existing_parent(tmp_path: Path) -> None:
    assert system_with().free_disk_mb(tmp_path / "missing" / "deeper") > 0


def test_signature_parses_powershell_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.dumps({"valid": True, "signer": "Ollama Inc."})
    monkeypatch.setattr(
        windows_ollama.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=payload, returncode=0),
    )
    assert system_with().signature(Path("x.exe")) == Signature(True, "Ollama Inc.")


def test_signature_unreadable_output_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        windows_ollama.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="", returncode=1)
    )
    assert system_with().signature(Path("x.exe")) == Signature(False, None)


def test_run_installer_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 3)

    monkeypatch.setattr(windows_ollama.subprocess, "run", fake_run)
    assert system_with().run_installer(Path("setup.exe"), ("/VERYSILENT",)) == 3
    assert captured == [["setup.exe", "/VERYSILENT"]]


def test_spawn_server_keeps_a_hidden_console_for_its_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DETACHED_PROCESS would drop the console, and each model runner would flash a window."""
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr(
        windows_ollama.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw))
    )
    system_with().spawn_server(Path("ollama.exe"))
    argv, kwargs = calls[0]
    assert argv == ["ollama.exe", "serve"]
    flags = kwargs["creationflags"]
    assert isinstance(flags, int)
    assert flags & 0x08000000  # CREATE_NO_WINDOW
    assert not flags & 0x00000008  # DETACHED_PROCESS


def test_signature_non_object_json_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        windows_ollama.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="[1]", returncode=0),
    )
    assert system_with().signature(Path("x.exe")) == Signature(False, None)


def test_signature_timeout_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("powershell", 1)

    monkeypatch.setattr(windows_ollama.subprocess, "run", hang)
    assert system_with().signature(Path("x.exe")) == Signature(False, None)


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell")
def test_signature_really_reads_a_signed_system_binary() -> None:
    binary = Path(os.environ["SYSTEMROOT"]) / "System32" / "cmd.exe"
    result = system_with().signature(binary)
    assert result.valid
    assert result.signer


def test_run_installer_timeout_is_a_failure_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("setup.exe", 1)

    monkeypatch.setattr(windows_ollama.subprocess, "run", hang)
    assert system_with().run_installer(Path("setup.exe"), ()) != 0
