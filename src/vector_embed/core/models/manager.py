"""Changing models from the app: pull, per-role overrides, and the guarded embedder switch.

The embedder is pinned because vectors from different models cannot be mixed: switching it
needs an explicit confirmation, rewrites ``settings.toml`` and schedules a full re-scan so the
worker rebuilds the index with the new model.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from vector_embed.core.models.catalog import ROLE_EMBED, ROLES
from vector_embed.core.models.registry import ModelRegistry, ReindexNotice
from vector_embed.core.providers.base import PullProgress
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings_io import set_setting
from vector_embed.core.store.sqlite import EMBEDDER_APPROVED_KEY, StateDb

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[PullProgress], None]


class ModelChangeError(RuntimeError):
    """A requested model change is not allowed (unknown role, missing confirmation)."""


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        provider: OllamaProvider,
        state: StateDb,
        settings_path: Path,
        indexed_files: Callable[[], int] = lambda: 0,
    ) -> None:
        self._registry = registry
        self._provider = provider
        self._state = state
        self._settings_path = settings_path
        self._indexed_files = indexed_files

    @property
    def registry(self) -> ModelRegistry:
        return self._registry

    def pull(self, name: str, progress: ProgressCallback | None = None) -> None:
        """Download ``name`` (the "better option available" one-click pull)."""
        for step in self._provider.pull(name):
            if progress is not None:
                progress(step)
        self._registry.refresh()

    def set_override(self, role: str, model: str | None) -> None:
        """Pin ``role`` to ``model`` (or clear the override). The embed role is not overridable."""
        if role not in ROLES:
            raise ModelChangeError(f"unknown role {role!r}")
        if role == ROLE_EMBED:
            raise ModelChangeError("change the embedder with change_embedder (it needs a re-index)")
        set_setting(self._settings_path, ["models", "overrides", role], model)
        self._registry.set_override(role, model)

    def embed_change_notice(self, new_model: str) -> ReindexNotice | None:
        """What switching the embedder costs; ``None`` if it is already the pinned model."""
        return self._registry.reindex_notice(new_model, self._indexed_files())

    def change_embedder(self, new_model: str, confirmed: bool = False) -> ReindexNotice | None:
        """Switch the embedder after explicit confirmation; returns the notice that was shown."""
        notice = self.embed_change_notice(new_model)
        if notice is None:
            return None
        if not confirmed:
            raise ModelChangeError(notice.message + " Confirm to continue.")
        set_setting(self._settings_path, ["embedding", "model"], new_model)
        self._registry.set_pinned_embed(new_model)
        self._state.set_meta(EMBEDDER_APPROVED_KEY, new_model)  # the only way the index is wiped
        # The worker wipes the old vectors on its next run; a due reconcile re-queues every file.
        self._state.set_meta("last_reconcile", "0")
        logger.info("embedder changed", extra={"model": new_model})
        return notice
