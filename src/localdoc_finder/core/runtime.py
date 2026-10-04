"""Composition root helpers: build the real collaborators from ``Settings``.

Front-ends (worker, watcher, CLI, app) call these instead of wiring objects themselves, so
tests can substitute any piece.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from localdoc_finder.core.cloud import CloudChatProvider, CloudConsent, CloudRouter
from localdoc_finder.core.doctypes.base import PROTOTYPE_TEXTS, build_prototypes
from localdoc_finder.core.documents import DocumentLoader
from localdoc_finder.core.extractors.base import ExtractContext, ExtractorSet
from localdoc_finder.core.extractors.image import OllamaCaptioner
from localdoc_finder.core.extractors.ocr import WindowsOcr
from localdoc_finder.core.llm import LlmGateway
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.registry import ModelRegistry
from localdoc_finder.core.power import PowerGate
from localdoc_finder.core.privacy.policy import DocTypeLookup, PrivacyFilter
from localdoc_finder.core.providers.ollama import OllamaProvider
from localdoc_finder.core.providers.openai_compat import OpenAICompatibleProvider
from localdoc_finder.core.scope import ScopePolicy
from localdoc_finder.core.secrets import KeyringStore, KeyStore, KeyStoreError
from localdoc_finder.core.settings import Settings
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.store.lance import LanceStore
from localdoc_finder.core.store.sqlite import EMBEDDER_APPROVED_KEY, StateDb
from localdoc_finder.core.wiring import LOGS_DIRNAME as LOGS_DIRNAME  # noqa: PLC0414
from localdoc_finder.core.wiring import THUMBS_DIRNAME as THUMBS_DIRNAME  # noqa: PLC0414
from localdoc_finder.core.wiring import build_projects as build_projects  # noqa: PLC0414
from localdoc_finder.core.wiring import build_scope as build_scope  # noqa: PLC0414
from localdoc_finder.core.wiring import log_dir as log_dir  # noqa: PLC0414

logger = logging.getLogger(__name__)

_PROTOTYPES_KEY = "doctype_prototypes"


def build_provider(settings: Settings) -> OllamaProvider:
    return OllamaProvider(settings.embedding, host=settings.ollama_host)


def build_extractors(
    settings: Settings, scope: ScopePolicy, provider: OllamaProvider | None = None
) -> ExtractorSet:
    """Extractors wired with Windows OCR and, when enabled, the vision captioner."""
    captioner = None
    if settings.images.enable_captions and provider is not None:
        captioner = OllamaCaptioner(
            provider.client, settings.images.caption_model, free_gpu=provider.unload_embedder
        )
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
    approved = state.get_meta(EMBEDDER_APPROVED_KEY) == model  # a confirmed embedder change
    store = LanceStore(
        settings.storage.data_dir,
        state,
        model,
        dim,
        vector_index_min_rows=settings.search.vector_index_min_rows,
        allow_wipe=approved,
    )
    if approved:
        state.delete_meta(EMBEDDER_APPROVED_KEY)  # one rebuild per confirmation
    return store


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
    *,
    doc_types: DocTypeLookup | None = None,
) -> CloudContext:
    """Cloud wiring. With no provider configured (the default) nothing can leave the machine."""
    privacy = PrivacyFilter(settings.privacy, scope, doc_types)
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
    cloud = build_cloud(settings, state, scope, registry, keys, doc_types=store.doc_types_for)
    if allow_cloud:
        gateway.router = cloud.router
    if settings.privacy.mask_ids_locally:
        gateway.local_filter = cloud.privacy.mask_ids
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
