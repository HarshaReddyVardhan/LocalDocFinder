from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.core.fakes import DIM, FakeCloudInner, FakeEmbedder
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed.core.cloud import CloudChatProvider, CloudConsent, CloudRouter
from vector_embed.core.doctypes.base import DocTypeClassifierSet
from vector_embed.core.documents import DocumentLoader
from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.indexer import Indexer
from vector_embed.core.llm import LlmGateway
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.power import PowerGate
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.projects import Projects
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.runtime import CloudContext
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import (
    CloudProviderSettings,
    CloudSettings,
    EmbeddingSettings,
    PowerSettings,
    ScopeSettings,
    Settings,
    StorageSettings,
)
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.store.lance import LanceStore
from vector_embed.core.store.sqlite import StateDb


@pytest.fixture
def scope_settings() -> ScopeSettings:
    """Defaults, minus ``appdata``: pytest's tmp dirs live under AppData on Windows."""
    default = ScopeSettings()
    return ScopeSettings(blocked_dirs=default.blocked_dirs - {"appdata"})


@pytest.fixture
def policy(scope_settings: ScopeSettings) -> ScopePolicy:
    return ScopePolicy(scope_settings)


@pytest.fixture
def make(tmp_path: Path) -> Callable[..., str]:
    """Create a non-empty text file under tmp_path and return its path."""

    def _make(rel: str, content: str = "x = 1\n") -> str:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    return _make


@dataclass
class Env:
    """A complete indexing stack on a temp directory with a fake embedder."""

    root: Path
    data_dir: Path
    settings: Settings
    state: StateDb
    store: LanceStore
    embedder: FakeEmbedder
    scope: ScopePolicy
    projects: Projects
    extractors: ExtractorSet
    classifier: DocTypeClassifierSet
    indexer: Indexer


@pytest.fixture
def env(tmp_path: Path, scope_settings: ScopeSettings) -> Iterator[Env]:
    root = tmp_path / "work"
    root.mkdir()
    data_dir = tmp_path / "data"
    settings = Settings(scope=scope_settings, storage=StorageSettings(data_dir=data_dir))
    scope = ScopePolicy(scope_settings, blocked_roots=[data_dir])
    state = StateDb(data_dir)
    store = LanceStore(data_dir, state, "fake-model", dim=DIM)
    embedder = FakeEmbedder()
    projects = Projects(scope, scope_settings, roots=[root])
    extractors = ExtractorSet(ExtractContext(scope, scope_settings, chunking=settings.chunking))
    classifier = DocTypeClassifierSet(settings.doctypes, scope_settings)
    indexer = Indexer(
        settings,
        state,
        store,
        embedder,
        extractors=extractors,
        projects=projects,
        scope=scope,
        classifier=classifier,
    )
    yield Env(
        root, data_dir, settings, state, store, embedder, scope, projects, extractors,
        classifier, indexer,
    )  # fmt: skip
    state.close()


@dataclass
class Power:
    """Mutable AC state shared with the PowerGate used by ``skill_ctx``."""

    on_ac: bool = True


@pytest.fixture
def power_state() -> Power:
    return Power()


@pytest.fixture
def skill_ctx(env: Env, power_state: Power) -> SkillContext:
    gate = PowerGate(PowerSettings(), probe=lambda: power_state.on_ac)
    gate.update()
    return SkillContext(env.settings, env.state, env.store, env.embedder, gate)


GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
CHAT_MODELS: dict[str, dict[str, object]] = {
    "qwen3-embedding:0.6b": {"caps": ["embedding"], "size": 600 * 1024**2},
    "qwen3.5:9b": {"caps": ["completion"], "size": 5 * 1024**3},
    "qwen2.5-coder:7b": {"caps": ["completion"], "size": 4 * 1024**3},
}


@dataclass
class Chat:
    """A chat stack on a fake Ollama client, attached to ``skill_ctx.extras``."""

    gateway: LlmGateway
    client: FakeOllamaClient
    provider: OllamaProvider

    def chat_calls(self) -> list[dict[str, object]]:
        return [kw for kind, kw in self.client.calls if kind == "chat"]

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.client.calls]


@pytest.fixture
def chat(skill_ctx: SkillContext, env: Env) -> Chat:
    client = FakeOllamaClient(models={k: dict(v) for k, v in CHAT_MODELS.items()})
    provider = OllamaProvider(EmbeddingSettings(), client=client, sleep=lambda _s: None)
    registry = ModelRegistry(
        load_catalog(),
        [provider],
        env.state,
        hardware_probe=lambda: GPU,
        pinned_embed=env.settings.embedding.model,
    )
    gateway = LlmGateway(env.settings.chat, registry, provider, env.state, skill_ctx.power)
    documents = DocumentLoader(env.store, env.scope, lambda: env.extractors)
    skill_ctx.extras.update(
        llm=gateway, provider=provider, models=registry, documents=documents, scope=env.scope
    )
    return Chat(gateway, client, provider)


@dataclass
class CloudRig:
    inner: "FakeCloudInner"
    provider: CloudChatProvider
    router: CloudRouter
    consent: CloudConsent
    privacy: PrivacyFilter


@pytest.fixture
def cloud(chat: Chat, env: Env, skill_ctx: SkillContext) -> CloudRig:
    """The chat stack with a scripted cloud provider attached (policy chosen per test)."""
    inner = FakeCloudInner()
    settings = CloudSettings(
        providers={
            "openrouter": CloudProviderSettings(
                base_url="https://openrouter.ai/api/v1",
                label="OpenRouter",
                models={"chat": "vendor/chat", "match_scorer": "vendor/judge"},
            )
        },
        active="openrouter",
        routing={"chat": "cloud", "match_scorer": "cloud", "code_chat": "cloud"},
    )
    privacy = PrivacyFilter(env.settings.privacy, env.scope)
    consent = CloudConsent()
    consent.grant()
    provider = CloudChatProvider(inner, privacy, env.state, settings, consent)
    router = CloudRouter(settings, provider, chat.gateway._registry)
    chat.gateway.router = router
    skill_ctx.extras.update(privacy=privacy, cloud=CloudContext(privacy, consent, router, provider))
    return CloudRig(inner, provider, router, consent, privacy)
