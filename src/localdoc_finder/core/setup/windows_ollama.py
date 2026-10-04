"""``OllamaSystem`` for Windows: real network, disk, process and signature calls."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from localdoc_finder.core.setup.ollama_install import ProgressCallback, Signature

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_CHUNK_BYTES = 256 * 1024
_PING_TIMEOUT_SECONDS = 2.0
_TIMEOUT_SECONDS = 60.0  # per read, so a big download is fine as long as bytes keep arriving
_SIGNATURE_TIMEOUT_SECONDS = 60
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_SIGNATURE_PATH_ENV = "LDF_SIGNATURE_PATH"
_INSTALLER_TIMEOUT_SECONDS = 15 * 60
_INSTALLER_TIMED_OUT = -1
# The path travels in the environment: with ``-Command`` extra argv entries are pasted into the
# script text, so ``$args[0]`` is empty and quoting a path with spaces is fragile.
_SIGNATURE_SCRIPT = (
    f"$s = Get-AuthenticodeSignature -LiteralPath $env:{_SIGNATURE_PATH_ENV}; "
    "[pscustomobject]@{valid = ($s.Status -eq 'Valid'); "
    "signer = if ($s.SignerCertificate) "
    "{ $s.SignerCertificate.GetNameInfo('SimpleName', $false) } else { $null }} "
    "| ConvertTo-Json -Compress"
)


class WindowsOllamaSystem:
    def __init__(
        self, host: str = "http://127.0.0.1:11434", client: httpx.Client | None = None
    ) -> None:
        self._host = host.rstrip("/")
        self._client = client or httpx.Client(
            follow_redirects=True, timeout=httpx.Timeout(_TIMEOUT_SECONDS)
        )

    def ping(self) -> bool:
        try:
            response = self._client.get(self._host + "/api/version", timeout=_PING_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return False
        return response.status_code == httpx.codes.OK

    def find_executable(self) -> Path | None:
        on_path = shutil.which("ollama")
        if on_path:
            return Path(on_path)
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.is_file():
                return candidate
        return None

    def download(self, url: str, destination: Path, progress: ProgressCallback) -> None:
        partial = destination.with_name(destination.name + ".part")
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                total = int(length) if length else None
                done = 0
                with partial.open("wb") as out:
                    for chunk in response.iter_bytes(_CHUNK_BYTES):
                        out.write(chunk)
                        done += len(chunk)
                        progress(done, total)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    def signature(self, path: Path) -> Signature:
        powershell = shutil.which("powershell") or "powershell"
        try:
            result = self._run_signature_script(powershell, path)
        except subprocess.TimeoutExpired:
            logger.warning("ollama: signature check timed out")
            return Signature(valid=False, signer=None)
        return self._parse_signature(result.stdout)

    @staticmethod
    def _run_signature_script(powershell: str, path: Path) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(  # noqa: S603  # fixed argv, the path travels in the environment
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _SIGNATURE_SCRIPT,
            ],
            env={**os.environ, _SIGNATURE_PATH_ENV: str(path)},
            capture_output=True,
            text=True,
            timeout=_SIGNATURE_TIMEOUT_SECONDS,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )

    @staticmethod
    def _parse_signature(stdout: str) -> Signature:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            logger.warning("ollama: could not read the installer signature")
            return Signature(valid=False, signer=None)
        return Signature(valid=bool(data.get("valid")), signer=data.get("signer"))

    def run_installer(self, path: Path, args: tuple[str, ...]) -> int:
        try:
            return subprocess.run(  # noqa: S603
                [str(path), *args], check=False, timeout=_INSTALLER_TIMEOUT_SECONDS
            ).returncode
        except subprocess.TimeoutExpired:
            logger.warning("ollama: installer timed out")
            return _INSTALLER_TIMED_OUT

    def spawn_server(self, executable: Path) -> None:
        # Not DETACHED_PROCESS: it overrides CREATE_NO_WINDOW and leaves the server with no console,
        # so every model runner it starts (llama-server.exe) would flash its own console window.
        # With a hidden console the runners inherit it. A new process group keeps Ollama running
        # after this process exits.
        subprocess.Popen(  # noqa: S603  # outlives this process on purpose
            [str(executable), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP,
        )

    def models_dir(self) -> Path:
        configured = os.environ.get("OLLAMA_MODELS")
        return Path(configured) if configured else Path.home() / ".ollama" / "models"

    def free_disk_mb(self, path: Path) -> int:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free // _MB
