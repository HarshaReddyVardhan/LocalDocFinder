"""Qt-free logic behind the Models tab and the health dashboard."""

from collections.abc import Callable
from pathlib import Path

from vector_embed.core.health import HealthSnapshot, collect_health, format_health
from vector_embed.core.models.catalog import ROLE_CHAT
from vector_embed.core.models.manager import ModelManager, ProgressCallback
from vector_embed.core.models.registry import ModelRegistry, ReindexNotice, Report
from vector_embed.core.models.starter import StarterPick, pick_starter
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.skills.base import SkillContext


class ModelsController:
    def __init__(self, context_factory: Callable[[], SkillContext], settings_path: Path) -> None:
        self._factory = context_factory
        self._settings_path = settings_path
        self._manager: ModelManager | None = None

    @property
    def manager(self) -> ModelManager:
        if self._manager is None:
            ctx = self._factory()
            provider = ctx.extras["provider"]
            registry = ctx.extras["models"]
            assert isinstance(provider, OllamaProvider)
            assert isinstance(registry, ModelRegistry)
            self._manager = ModelManager(
                registry, provider, ctx.state, self._settings_path, ctx.state.manifest_count
            )
        return self._manager

    @property
    def registry(self) -> ModelRegistry:
        return self.manager.registry

    def report(self) -> Report:
        """Re-discover installed models and report roles, flags and recommendations."""
        self.registry.refresh()
        return self.registry.report()

    def health(self) -> HealthSnapshot:
        ctx = self._factory()
        provider = ctx.extras["provider"]
        assert isinstance(provider, OllamaProvider)
        return collect_health(
            ctx.state,
            ctx.store,
            self.registry,
            provider.loaded_models,
            budget_usd=ctx.settings.cloud.monthly_budget_usd,
        )

    def health_text(self) -> str:
        return format_health(self.health())

    def missing_chat_model(self) -> StarterPick | None:
        """The chat model Ask, Chat and Match need, if none is installed and one fits."""
        report = self.report()
        if report.resolutions[ROLE_CHAT].model is not None:
            return None
        starter = pick_starter(self.registry.catalog, report.hardware, (ROLE_CHAT,))
        return starter.pick_for(ROLE_CHAT)

    def pull(self, name: str, progress: ProgressCallback | None = None) -> None:
        self.manager.pull(name, progress)

    def set_override(self, role: str, model: str | None) -> None:
        self.manager.set_override(role, model)

    def embed_notice(self, model: str) -> ReindexNotice | None:
        return self.manager.embed_change_notice(model)

    def change_embedder(self, model: str) -> None:
        self.manager.change_embedder(model, confirmed=True)
