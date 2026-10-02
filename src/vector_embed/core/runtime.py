"""Composition root helpers: build the real collaborators from ``Settings``.

Front-ends (worker, watcher, CLI, app) call these instead of wiring objects themselves, so
tests can substitute any piece.
"""

import json
import logging
from pathlib import Path

import numpy as np

from vector_embed.core.doctypes.base import PROTOTYPE_TEXTS, build_prototypes
from vector_embed.core.extractors.base import ExtractContext, ExtractorSet
from vector_embed.core.extractors.image import OllamaCaptioner
from vector_embed.core.extractors.ocr import WindowsOcr
from vector_embed.core.projects import Projects
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.scope import ScopePolicy
from vector_embed.core.settings import Settings
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
