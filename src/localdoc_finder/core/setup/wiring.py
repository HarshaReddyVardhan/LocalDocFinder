"""Builds a ``SetupFlow`` wired to the real Ollama, hardware and settings.

The CLI and the app's setup wizard both use this; tests inject their own builder instead.
"""

from collections.abc import Callable
from typing import Protocol

from localdoc_finder.core import runtime
from localdoc_finder.core.models.benchmark import BenchResult, bench_chat, bench_embed
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.hardware import probe_hardware
from localdoc_finder.core.providers.ollama import OllamaProvider
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.setup.flow import SetupEvent, SetupFlow, SlowOffer
from localdoc_finder.core.setup.ollama_install import OllamaSetup
from localdoc_finder.core.setup.windows_ollama import WindowsOllamaSystem
from localdoc_finder.core.store.sqlite import StateDb


class FlowBuilder(Protocol):
    def __call__(
        self,
        settings: Settings,
        state: StateDb,
        progress: Callable[[SetupEvent], None],
        accept_downgrade: Callable[[SlowOffer], bool],
    ) -> SetupFlow: ...


def build_flow(
    settings: Settings,
    state: StateDb,
    progress: Callable[[SetupEvent], None],
    accept_downgrade: Callable[[SlowOffer], bool],
) -> SetupFlow:
    provider = runtime.build_provider(settings)

    def embed_bench(model: str) -> BenchResult:
        tuned = settings.embedding.model_copy(update={"model": model})
        return bench_embed(OllamaProvider(tuned, client=provider.client))

    return SetupFlow(
        ollama=OllamaSetup(WindowsOllamaSystem(settings.ollama_host)),
        host=provider,
        catalog=load_catalog(settings.storage.data_dir),
        hardware=probe_hardware(),
        state=state,
        settings_path=settings.settings_path(),
        data_dir=settings.storage.data_dir,
        current_embed=settings.embedding.model,
        bench_chat=lambda model: bench_chat(provider, model),
        bench_embed=embed_bench,
        progress=progress,
        accept_downgrade=accept_downgrade,
    )
