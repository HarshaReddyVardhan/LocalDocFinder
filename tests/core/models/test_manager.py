import tomllib
from pathlib import Path

import pytest
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.manager import ModelChangeError, ModelManager
from vector_embed.core.models.registry import ModelRegistry
from vector_embed.core.providers.base import PullProgress
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.settings import EmbeddingSettings
from vector_embed.core.store.sqlite import EMBEDDER_APPROVED_KEY, StateDb

GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)


class World:
    def __init__(self, tmp_path: Path) -> None:
        self.client = FakeOllamaClient(
            models={"qwen3-embedding:0.6b": {"caps": ["embedding"]}, "llama3.2": {}}
        )
        self.provider = OllamaProvider(EmbeddingSettings(), client=self.client)
        self.state = StateDb(tmp_path / "data")
        self.path = tmp_path / "data" / "settings.toml"
        self.registry = ModelRegistry(
            load_catalog(),
            [self.provider],
            self.state,
            hardware_probe=lambda: GPU,
            pinned_embed="qwen3-embedding:0.6b",
        )
        self.registry.refresh()
        self.manager = ModelManager(
            self.registry, self.provider, self.state, self.path, indexed_files=lambda: 1200
        )

    def settings(self) -> dict[str, object]:
        with self.path.open("rb") as handle:
            return tomllib.load(handle)


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.chdir(tmp_path)
    return World(tmp_path)


def test_pull_reports_progress_and_refreshes_the_model_list(world: World) -> None:
    seen: list[PullProgress] = []
    world.client.models["qwen3.5:9b"] = {}
    world.manager.pull("qwen3.5:9b", seen.append)
    assert seen[0].fraction == 0.5
    assert [c[0] for c in world.client.calls].count("pull") == 1
    assert "qwen3.5:9b" in {m.name for m in world.registry.installed}
    world.manager.pull("qwen3.5:9b")  # progress callback is optional


def test_role_override_is_written_to_settings(world: World) -> None:
    world.manager.set_override("chat", "llama3.2")
    assert world.settings()["models"] == {"overrides": {"chat": "llama3.2"}}
    world.manager.set_override("chat", None)
    assert world.settings()["models"] == {"overrides": {}}


def test_unknown_roles_and_the_embed_role_are_refused(world: World) -> None:
    with pytest.raises(ModelChangeError, match="unknown role"):
        world.manager.set_override("poetry", "x")
    with pytest.raises(ModelChangeError, match="change_embedder"):
        world.manager.set_override("embed", "bge-m3")


def test_embedder_change_needs_confirmation_and_describes_the_cost(world: World) -> None:
    notice = world.manager.embed_change_notice("bge-m3")
    assert notice is not None
    assert notice.files == 1200
    with pytest.raises(ModelChangeError, match="re-indexing ~1200 files"):
        world.manager.change_embedder("bge-m3")
    assert not world.path.exists()  # nothing was written without confirmation


def test_confirmed_embedder_change_rewrites_settings_and_schedules_a_rescan(
    world: World,
) -> None:
    world.state.set_meta("last_reconcile", "123456")
    notice = world.manager.change_embedder("bge-m3", confirmed=True)
    assert notice is not None
    assert world.settings()["embedding"] == {"model": "bge-m3"}
    assert world.state.get_meta("last_reconcile") == "0"
    assert (
        world.state.get_meta(EMBEDDER_APPROVED_KEY) == "bge-m3"
    )  # the one thing that allows a wipe


def test_an_unconfirmed_change_leaves_no_approval(world: World) -> None:
    with pytest.raises(ModelChangeError):
        world.manager.change_embedder("bge-m3")
    assert world.state.get_meta(EMBEDDER_APPROVED_KEY) is None


def test_selecting_the_current_embedder_is_not_a_change(world: World) -> None:
    assert world.manager.embed_change_notice("qwen3-embedding:0.6b") is None
    assert world.manager.change_embedder("qwen3-embedding:0.6b", confirmed=True) is None
    assert not world.path.exists()


def test_overrides_and_the_pinned_embedder_take_effect_in_the_running_registry(
    world: World,
) -> None:
    world.manager.set_override("chat", "llama3.2")
    assert world.registry.resolve("chat").model == "llama3.2"
    assert world.registry.resolve("chat").reason == "override"
    world.manager.set_override("chat", None)
    assert world.registry.resolve("chat").reason != "override"
    world.client.models["bge-m3"] = {"caps": ["embedding"]}
    world.registry.refresh()
    world.manager.change_embedder("bge-m3", confirmed=True)
    assert world.registry.resolve("embed").model == "bge-m3"
    assert world.registry.resolve("embed").reason == "pinned"
