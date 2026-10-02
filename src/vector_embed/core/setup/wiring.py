"""Builds a ``SetupFlow`` wired to the real Ollama, hardware and settings.

The CLI and the app's setup wizard both use this; tests inject their own builder instead.
"""

from collections.abc import Callable
from typing import Protocol

from vector_embed.core import runtime
from vector_embed.core.models.benchmark import BenchResult, bench_chat, bench_embed
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import probe_hardware
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings import SETTINGS_FILENAME, Settings
from vector_embed.core.setup.flow import SetupEvent, SetupFlow, SlowOffer
from vector_embed.core.setup.ollama_install import OllamaSetup
from vector_embed.core.setup.windows_ollama import WindowsOllamaSystem
from vector_embed.core.store.sqlite import StateDb


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
        settings_path=settings.storage.data_dir / SETTINGS_FILENAME,
        data_dir=settings.storage.data_dir,
        current_embed=settings.embedding.model,
        bench_chat=lambda model: bench_chat(provider, model),
        bench_embed=embed_bench,
        progress=progress,
        accept_downgrade=accept_downgrade,
    )
