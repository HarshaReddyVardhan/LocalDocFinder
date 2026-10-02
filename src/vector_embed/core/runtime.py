"""Composition root helpers: build the real collaborators from ``Settings``.

Front-ends (worker, watcher, CLI, app) call these instead of wiring objects themselves, so
tests can substitute any piece.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vector_embed.core.cloud import CloudChatProvider, CloudConsent, CloudRouter
from vector_embed.core.doctypes.base import PROTOTYPE_TEXTS, build_prototypes
from vector_embed.core.documents import DocumentLoader
from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.extractors.image import OllamaCaptioner
from vector_embed.core.extractors.ocr import WindowsOcr
from vector_embed.core.llm import LlmGateway
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.power import PowerGate
from vector_embed.core.privacy.policy import PrivacyFilter
from vector_embed.core.projects import Projects
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.providers.openai_compat import OpenAICompatibleProvider
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.secrets import KeyringStore, KeyStore, KeyStoreError
from vector_embed.core.settings import Settings
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.store.lance import LanceStore
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger(__name__)

LOGS_DIRNAME = "logs"
THUMBS_DIRNAME = "thumbs"
_PROTOTYPES_KEY = "doctype_prototypes"


def build_scope(settings: Settings) -> ScopePolicy:
    """The scope policy; the app's own data folder is never indexed."""
    return ScopePolicy(settings.scope, blocked_roots=[settings.storage.data_dir])


def build_projects(settings: Settings, scope: ScopePolicy) -> Projects:
    return Projects(scope, settings.scope)


def build_provider(settings: Settings) -> OllamaProvider:
    return OllamaProvider(settings.embedding, host=settings.ollama_host)


def build_extractors(
    settings: Settings, scope: ScopePolicy, provider: OllamaProvider | None = None
) -> ExtractorSet:
    """Extractors wired with Windows OCR and, when enabled, the vision captioner."""
    captioner = None
    if settings.images.enable_captions and provider is not None:
        captioner = OllamaCaptioner(provider.client, settings.images.caption_model)
    ctx = ExtractContext(
        scope=scope,
        scope_settings=settings.scope,
        chunking=settings.chunking,
        images=settings.images,
        ocr=WindowsOcr(settings.images.ocr_max_dimension),
        captioner=captioner,
        thumbs_dir=settings.storage.data_dir / THUMBS_DIRNAME,
    )
    return ExtractorSet(ctx)


def open_store(settings: Settings, state: StateDb, provider: OllamaProvider) -> LanceStore:
    """Open the vector store, probing the embedding dimension only when the model changed."""
    model = settings.embedding.model
    stored_model, stored_dim = state.get_meta("model_id"), state.get_meta("dim")
    dim = int(stored_dim) if stored_model == model and stored_dim else provider.dim
    return LanceStore(
        settings.storage.data_dir,
        state,
        model,
        dim,
        vector_index_min_rows=settings.search.vector_index_min_rows,
    )


def open_read_only_store(settings: Settings, state: StateDb) -> LanceStore:
    """Store for searching: opens whatever tables exist, never creates or wipes anything."""
    return LanceStore(settings.storage.data_dir, state, settings.embedding.model, dim=None)


def load_prototypes(state: StateDb, provider: OllamaProvider) -> dict[str, np.ndarray]:
    """Doctype prototype vectors, cached per embedding model so they are embedded once."""
    model = provider.embed_model
    cached = state.get_meta(_PROTOTYPES_KEY)
    if cached:
        data = json.loads(cached)
        if data.get("model") == model and set(data["vectors"]) == set(PROTOTYPE_TEXTS):
            return {k: np.asarray(v, dtype=np.float32) for k, v in data["vectors"].items()}
    prototypes = build_prototypes(lambda texts: provider.embed(texts, kind="doc"))
    state.set_meta(
        _PROTOTYPES_KEY,
        json.dumps({"model": model, "vectors": {k: v.tolist() for k, v in prototypes.items()}}),
    )
    return prototypes


def log_dir(settings: Settings) -> Path:
    return settings.storage.data_dir / LOGS_DIRNAME


def build_model_registry(
    settings: Settings, state: StateDb, provider: OllamaProvider
) -> ModelRegistry:
    return ModelRegistry(
        load_catalog(settings.storage.data_dir),
        [provider],
        state,
        overrides=settings.models.overrides,
        pinned_embed=settings.embedding.model,
        resident_vram_mb=provider.resident_vram_mb,
    )


@dataclass
class CloudContext:
    """Everything cloud-related for one process: the privacy filter, consent and routing."""

    privacy: PrivacyFilter
    consent: CloudConsent
    router: CloudRouter
    provider: CloudChatProvider | None


def build_cloud(
    settings: Settings,
    state: StateDb,
    scope: ScopePolicy,
    registry: ModelRegistry,
    keys: KeyStore | None = None,
) -> CloudContext:
    """Cloud wiring. With no provider configured (the default) nothing can leave the machine."""
    privacy = PrivacyFilter(settings.privacy, scope)
    consent = CloudConsent()
    provider: CloudChatProvider | None = None
    active = settings.cloud.active
    if active and active in settings.cloud.providers:
        try:
            key = (keys or KeyringStore()).get(active)
        except KeyStoreError:
            logger.warning("cloud: cannot read the API key for %s", active, exc_info=True)
            key = None
        if key:
            inner = OpenAICompatibleProvider(active, settings.cloud.providers[active], key)
            provider = CloudChatProvider(inner, privacy, state, settings.cloud, consent)
        else:
            logger.info("cloud: no API key stored for %s; staying local", active)
    return CloudContext(privacy, consent, CloudRouter(settings.cloud, provider, registry), provider)


def build_skill_context(
    settings: Settings,
    state: StateDb,
    fullscreen: Callable[[], bool] = lambda: False,
    keys: KeyStore | None = None,
    *,
    owner: str = "app",
    allow_cloud: bool = True,
) -> SkillContext:
    """Context for read-side skills (search, ask, ...): read-only store, local provider.

    ``owner`` names this process in the chat lock; ``allow_cloud=False`` keeps every model call
    on the local machine (used by front-ends whose caller is not the user, such as MCP).
    """
    provider = build_provider(settings)
    power = PowerGate(settings.power)
    power.update()
    registry = build_model_registry(settings, state, provider)
    gateway = LlmGateway(
        settings.chat,
        registry,
        provider,
        state,
        power,
        fullscreen=fullscreen,
        owner=owner,
        refresh_seconds=settings.models.refresh_seconds,
    )
    store = open_read_only_store(settings, state)
    scope = build_scope(settings)
    cloud = build_cloud(settings, state, scope, registry, keys)
    if allow_cloud:
        gateway.router = cloud.router
    cache: list[ExtractorSet] = []

    def extractors() -> ExtractorSet:  # built on first use: OCR setup is not free
        if not cache:
            cache.append(build_extractors(settings, scope, provider))
        return cache[0]

    return SkillContext(
        settings=settings,
        state=state,
        store=store,
        embedder=provider,
        power=power,
        extras={
            "provider": provider,
            "models": registry,
            "llm": gateway,
            "scope": scope,
            "privacy": cloud.privacy,
            "cloud": cloud,
            "documents": DocumentLoader(store, scope, extractors),
        },
    )
