"""Detect, download, verify, install and start Ollama.

Decision logic only; every touch of the machine (network, disk, processes, signatures) goes
through the ``OllamaSystem`` protocol so tests fake it. Nothing is downloaded or installed
without the caller passing explicit consent, and the installer's Authenticode signature is
checked before it runs.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
INSTALLER_NAME = "OllamaSetup.exe"
INSTALLER_SIZE_MB = 1200  # approximate; shown to the user before they consent
INSTALLED_SIZE_MB = 4500  # approximate: the unpacked program with its GPU runtimes
EXPECTED_SIGNER = "Ollama Inc."
INSTALLER_ARGS = ("/CURRENTUSER", "/SP-", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART")
SERVER_WAIT_SECONDS = 60.0
SERVER_POLL_SECONDS = 1.0

# (bytes downloaded so far, total bytes or None when the server did not say)
ProgressCallback = Callable[[int, int | None], None]


class OllamaState(StrEnum):
    RUNNING = "running"
    INSTALLED_NOT_RUNNING = "installed_not_running"
    MISSING = "missing"


class OllamaSetupError(RuntimeError):
    """Installing or starting Ollama failed; the message is safe to show to the user."""


@dataclass(frozen=True)
class Signature:
    valid: bool
    signer: str | None  # common name of the signing certificate


class OllamaSystem(Protocol):
    """Everything the installer needs from the machine."""

    def ping(self) -> bool: ...
    def find_executable(self) -> Path | None: ...
    def download(self, url: str, destination: Path, progress: ProgressCallback) -> None: ...
    def signature(self, path: Path) -> Signature: ...
    def run_installer(self, path: Path, args: tuple[str, ...]) -> int: ...
    def spawn_server(self, executable: Path) -> None: ...
    def models_dir(self) -> Path: ...
    def free_disk_mb(self, path: Path) -> int: ...


class OllamaSetup:
    def __init__(
        self,
        system: OllamaSystem,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        server_wait_seconds: float = SERVER_WAIT_SECONDS,
    ) -> None:
        self._system = system
        self._sleep = sleep
        self._clock = clock
        self._server_wait = server_wait_seconds

    def detect(self) -> OllamaState:
        """Pings the server first, then looks for an installed executable."""
        if self._system.ping():
            return OllamaState.RUNNING
        if self._system.find_executable() is not None:
            return OllamaState.INSTALLED_NOT_RUNNING
        return OllamaState.MISSING

    def free_disk_mb(self) -> int:
        """Free space on the drive that holds (or will hold) the Ollama model store."""
        return self._system.free_disk_mb(self._system.models_dir())

    def free_disk_mb_at(self, path: Path) -> int:
        """Free space on the drive that holds ``path`` (or its nearest existing parent)."""
        return self._system.free_disk_mb(path)

    def download_installer(self, directory: Path, progress: ProgressCallback) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / INSTALLER_NAME
        self._system.download(INSTALLER_URL, target, progress)
        return target

    def verify_signature(self, path: Path) -> None:
        signature = self._system.signature(path)
        if not signature.valid:
            raise OllamaSetupError("The Ollama installer's signature is not valid; not running it.")
        if signature.signer != EXPECTED_SIGNER:
            raise OllamaSetupError(
                f"The Ollama installer is signed by {signature.signer!r}, "
                f"expected {EXPECTED_SIGNER!r}; not running it."
            )

    def run_installer(self, path: Path) -> None:
        """Run the verified installer silently, then wait for the server to answer."""
        code = self._system.run_installer(path, INSTALLER_ARGS)
        if code != 0:
            raise OllamaSetupError(f"The Ollama installer failed (exit code {code}).")
        self._wait_until_up()

    def start_server(self) -> None:
        """Launch an installed Ollama and wait for it; a no-op when it already answers."""
        if self._system.ping():
            return
        executable = self._system.find_executable()
        if executable is None:
            raise OllamaSetupError("Ollama is not installed.")
        self._system.spawn_server(executable)
        self._wait_until_up()

    def install(
        self,
        directory: Path,
        progress: ProgressCallback,
        *,
        consented: bool,
        checkpoint: Callable[[], None] = lambda: None,
    ) -> None:
        """Download, verify and run the official installer. Requires explicit consent.

        ``checkpoint`` is called between the phases; it may raise to abandon the install (the
        user closed the wizard). A running installer itself cannot be interrupted.
        """
        if not consented:
            raise OllamaSetupError("Installing Ollama needs the user's consent.")
        installer = directory / INSTALLER_NAME
        try:  # also covers the download: an interrupted or failed one must not leave a file behind
            self.download_installer(directory, progress)
            checkpoint()
            self.verify_signature(installer)
            checkpoint()
            self.run_installer(installer)
        finally:
            installer.unlink(missing_ok=True)

    def _wait_until_up(self) -> None:
        deadline = self._clock() + self._server_wait
        while not self._system.ping():
            if self._clock() >= deadline:
                raise OllamaSetupError("Ollama did not start in time.")
            self._sleep(SERVER_POLL_SECONDS)
        logger.info("ollama: server is up")
