"""``OllamaSystem`` for Windows: real network, disk, process and signature calls."""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import httpx

from vector_embed.core.setup.ollama_install import ProgressCallback, Signature

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
_CHUNK_BYTES = 256 * 1024
_PING_TIMEOUT_SECONDS = 2.0
_TIMEOUT_SECONDS = 60.0  # per read, so a big download is fine as long as bytes keep arriving
_SIGNATURE_TIMEOUT_SECONDS = 60
_DETACHED_PROCESS = 0x00000008
_CREATE_NO_WINDOW = 0x08000000
_SIGNATURE_SCRIPT = (
    "$s = Get-AuthenticodeSignature -LiteralPath $args[0]; "
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
        result = subprocess.run(  # noqa: S603  # fixed argv, the path is a separate argument
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _SIGNATURE_SCRIPT,
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=_SIGNATURE_TIMEOUT_SECONDS,
            creationflags=_CREATE_NO_WINDOW,
            check=False,
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("ollama: could not read the installer signature")
            return Signature(valid=False, signer=None)
        return Signature(valid=bool(data.get("valid")), signer=data.get("signer"))

    def run_installer(self, path: Path, args: tuple[str, ...]) -> int:
        return subprocess.run([str(path), *args], check=False).returncode  # noqa: S603

    def spawn_server(self, executable: Path) -> None:
        subprocess.Popen(  # noqa: S603  # detached on purpose: Ollama outlives this process
            [str(executable), "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_DETACHED_PROCESS | _CREATE_NO_WINDOW,
        )

    def models_dir(self) -> Path:
        configured = os.environ.get("OLLAMA_MODELS")
        return Path(configured) if configured else Path.home() / ".ollama" / "models"

    def free_disk_mb(self, path: Path) -> int:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return shutil.disk_usage(probe).free // _MB
