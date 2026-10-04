"""Shared fakes for the setup tests."""

import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

from localdoc_finder.core.models.benchmark import BenchKind, BenchmarkError, BenchResult
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.providers.base import ModelInfo, ProviderError, PullProgress
from localdoc_finder.core.setup.flow import SetupEvent, SetupFlow, SlowOffer
from localdoc_finder.core.setup.ollama_install import (
    EXPECTED_SIGNER,
    OllamaSetup,
    ProgressCallback,
    Signature,
)
from localdoc_finder.core.store.sqlite import StateDb


class FakeSystem:
    def __init__(self) -> None:
        self.up = False
        self.exe: Path | None = None
        self.sig = Signature(valid=True, signer=EXPECTED_SIGNER)
        self.installer_exit = 0
        self.up_after_pings: int | None = None  # server answers after this many failed pings
        self.pings = 0
        self.calls: list[str] = []
        self.free_mb = 50_000
        self.queried: list[Path] = []

    def ping(self) -> bool:
        self.pings += 1
        if self.up_after_pings is not None and self.pings > self.up_after_pings:
            self.up = True
        return self.up

    def find_executable(self) -> Path | None:
        return self.exe

    def download(self, url: str, destination: Path, progress: ProgressCallback) -> None:
        self.calls.append(f"download {url}")
        destination.write_bytes(b"MZ")
        progress(2, 2)

    def signature(self, path: Path) -> Signature:
        self.calls.append("signature")
        return self.sig

    def run_installer(self, path: Path, args: tuple[str, ...]) -> int:
        self.calls.append(f"run {args}")
        if self.installer_exit == 0:
            self.exe = Path("ollama.exe")
            self.up = self.up_after_pings is None
        return self.installer_exit

    def spawn_server(self, executable: Path) -> None:
        self.calls.append("spawn")
        self.up = self.up_after_pings is None

    def models_dir(self) -> Path:
        return Path("models")

    def free_disk_mb(self, path: Path) -> int:
        self.queried.append(path)
        return self.free_mb


class Clock:
    """Fake time: ``sleep`` advances it, so wait loops finish instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


GPU8 = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
FAST = {BenchKind.CHAT: 30.0, BenchKind.EMBED: 100.0}


class FakeHost:
    def __init__(self, installed: list[str] | None = None) -> None:
        self.installed = list(installed or [])
        self.pulled: list[str] = []
        self.fail_on: str | None = None

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name, "ollama") for name in self.installed]

    def pull(self, model: str) -> Iterator[PullProgress]:
        if model == self.fail_on:
            raise ProviderError("network down")
        yield PullProgress("pulling", 50, 100)
        self.pulled.append(model)
        self.installed.append(model)
        yield PullProgress("success")


class Harness:
    def __init__(self, tmp_path: Path, *, up: bool = True, installed: list[str] | None = None):
        self.system = FakeSystem()
        self.system.up = up
        if up:
            self.system.exe = Path("ollama.exe")
        clock = Clock()
        self.ollama = OllamaSetup(self.system, sleep=clock.sleep, clock=clock)
        self.host = FakeHost(installed)
        self.state = StateDb(tmp_path / "data")
        self.settings_path = tmp_path / "data" / "settings.toml"
        self.events: list[SetupEvent] = []
        self.rates = dict(FAST)
        self.model_rates: dict[str, float] = {}  # per-model override of ``rates``
        self.benched: list[str] = []
        self.offers: list[SlowOffer] = []
        self.accept = False
        self.bench_error: BenchmarkError | None = None
        self.tmp_path = tmp_path

    def bench(self, kind: BenchKind) -> Callable[[str], BenchResult]:
        def run(model: str) -> BenchResult:
            self.benched.append(model)
            if self.bench_error:
                raise self.bench_error
            return BenchResult(model, kind, self.model_rates.get(model, self.rates[kind]), 1.0)

        return run

    def flow(
        self,
        hardware: Hardware = GPU8,
        current_embed: str | None = None,
        progress: Callable[[SetupEvent], None] | None = None,
        accept: Callable[[SlowOffer], bool] | None = None,
    ) -> SetupFlow:
        def decide(offer: SlowOffer) -> bool:
            self.offers.append(offer)
            return accept(offer) if accept else self.accept

        return SetupFlow(
            ollama=self.ollama,
            host=self.host,
            catalog=load_catalog(),
            hardware=hardware,
            state=self.state,
            settings_path=self.settings_path,
            data_dir=self.tmp_path / "data",
            current_embed=current_embed,
            bench_chat=self.bench(BenchKind.CHAT),
            bench_embed=self.bench(BenchKind.EMBED),
            progress=progress or self.events.append,
            accept_downgrade=decide,
            clock=lambda: 1234.0,
        )

    def saved(self) -> dict[str, object]:
        with self.settings_path.open("rb") as handle:
            return tomllib.load(handle)
