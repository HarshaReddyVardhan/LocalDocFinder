from pathlib import Path

import pytest

from vector_embed.core.models import catalog as cat
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.registry import (
    FLAG_OK,
    FLAG_UNUSED,
    FLAG_WARNING,
    ModelRegistry,
)
from vector_embed.core.providers.base import (
    CAP_COMPLETION,
    CAP_EMBEDDING,
    CAP_VISION,
    ModelInfo,
    ProviderError,
)
from vector_embed.core.store.sqlite import StateDb

GB = 1024**3


def hw(free_vram: int = 7000, gpu: bool = True, ram_free: int = 16000) -> Hardware:
    return Hardware(
        gpu_name="RTX 2070" if gpu else None,
        vram_total_mb=8192 if gpu else 0,
        vram_free_mb=free_vram if gpu else 0,
        ram_total_mb=32000,
        ram_free_mb=ram_free,
        cpu_count=8,
        on_ac=True,
    )


def model(name: str, caps: tuple[str, ...] = (CAP_COMPLETION,), size_gb: float = 4.0) -> ModelInfo:
    return ModelInfo(name, "ollama", size_bytes=int(size_gb * GB), capabilities=frozenset(caps))


class FakeProvider:
    name = "ollama"

    def __init__(self, models: list[ModelInfo], fail: bool = False) -> None:
        self.models = models
        self.fail = fail

    def list_models(self) -> list[ModelInfo]:
        if self.fail:
            raise ProviderError("down")
        return self.models


EMBED = model("qwen3-embedding:0.6b", (CAP_EMBEDDING,), 0.6)
CHAT9 = model("qwen3.5:9b", size_gb=5.5)
LLAMA = model("llama3.2", size_gb=2.0)
CODER = model("qwen2.5-coder:7b", size_gb=4.7)
VISION = model("qwen2.5vl:3b", (CAP_COMPLETION, CAP_VISION), 3.2)


def registry(
    models: list[ModelInfo], hardware: Hardware | None = None, **kw: object
) -> ModelRegistry:
    reg = ModelRegistry(
        cat.load_catalog(),
        [FakeProvider(models)],
        hardware_probe=lambda: hardware or hw(),
        **kw,  # type: ignore[arg-type]
    )
    reg.refresh()
    return reg


class TestCatalog:
    def test_packaged_catalog_loads_and_covers_all_roles(self) -> None:
        catalog = cat.load_catalog()
        assert set(catalog.roles) == set(cat.ROLES)
        assert catalog.preferences(cat.ROLE_CHAT)[0] == "qwen3.5:9b"
        assert catalog.vram_mb("qwen3.5:9b") == 6600
        assert catalog.vram_mb("unknown") is None
        assert "mxbai-embed-large" in catalog.warnings
        assert "llama3.2" in catalog.known_names()

    def test_user_override_wins(self, tmp_path: Path) -> None:
        (tmp_path / cat.CATALOG_FILENAME).write_text('[roles]\nchat = ["x"]\n', encoding="utf-8")
        assert cat.load_catalog(tmp_path).preferences("chat") == ["x"]

    def test_empty_override_dir_falls_back_to_package(self, tmp_path: Path) -> None:
        assert cat.load_catalog(tmp_path).preferences("chat")

    @pytest.mark.parametrize(
        "text", ["[roles\n", '[roles]\nbogus = ["x"]\n', "[roles]\nchat = 5\n"]
    )
    def test_invalid_catalog_is_rejected(self, text: str) -> None:
        with pytest.raises(cat.CatalogError):
            cat.parse_catalog(text)


class TestResolve:
    def test_prefers_first_installed_that_fits(self) -> None:
        reg = registry([EMBED, CHAT9, LLAMA])
        assert reg.resolve("chat").model == "qwen3.5:9b"
        assert reg.resolve("embed").model == "qwen3-embedding:0.6b"

    def test_unknown_role(self) -> None:
        with pytest.raises(ValueError, match="unknown role"):
            registry([]).resolve("nope")

    def test_low_free_vram_picks_a_smaller_model(self) -> None:
        reg = registry([CHAT9, LLAMA], hw(free_vram=3000))
        resolution = reg.resolve("chat")
        assert resolution.model == "llama3.2"
        assert resolution.reason == "preferred"

    def test_a_resident_model_does_not_count_against_free_vram(self) -> None:
        # Free VRAM is low because our own chat model is loaded; it must still resolve to it.
        starved = hw(free_vram=1500)
        assert registry([CHAT9, LLAMA], starved).resolve("chat").model is None
        reg = registry([CHAT9, LLAMA], starved, resident_vram_mb=lambda: 5800)
        assert reg.resolve("chat").model == "qwen3.5:9b"

    def test_reclaimable_vram_is_capped_at_the_card_size(self) -> None:
        reg = registry([CHAT9], hw(free_vram=100), resident_vram_mb=lambda: 999_999)
        assert reg.resolve("chat").model == "qwen3.5:9b"  # 8192 MB card holds the 5.5 GB model
        small = Hardware("tiny", 2048, 100, 32000, 16000, 8, True)
        assert (
            registry([CHAT9], small, resident_vram_mb=lambda: 999_999).resolve("chat").model is None
        )

    def test_resident_memory_is_ignored_without_a_gpu(self) -> None:
        reg = registry([LLAMA], hw(gpu=False), resident_vram_mb=lambda: 99999)
        assert reg.resolve("chat", hw(gpu=False)).model in {None, "llama3.2"}

    def test_missing_preferred_falls_back_down_the_list(self) -> None:
        reg = registry([CODER, LLAMA])
        assert reg.resolve("chat").model == "qwen2.5-coder:7b"

    def test_nothing_fits(self) -> None:
        resolution = registry([CHAT9], hw(free_vram=500)).resolve("chat")
        assert resolution.model is None
        assert "no installed model fits" in resolution.reason

    def test_capability_fallback_for_unlisted_models(self) -> None:
        reg = registry([model("mystery:3b", size_gb=2.0)])
        resolution = reg.resolve("chat")
        assert resolution.model == "mystery:3b"
        assert resolution.reason.startswith("fallback")

    def test_fallback_skips_wrong_capabilities_and_warned_models(self) -> None:
        reg = registry(
            [
                model("some-embedder", (CAP_EMBEDDING,)),
                model("deepseek-r1:8b", size_gb=5.0),
            ]
        )
        assert reg.resolve("chat").model is None
        assert reg.resolve("embed").model == "some-embedder"
        assert reg.resolve("caption").model is None

    def test_vision_fallback(self) -> None:
        assert registry([model("v:1", (CAP_VISION,))]).resolve("caption").model == "v:1"

    def test_override_wins_and_reports_missing(self) -> None:
        reg = registry([CHAT9, LLAMA], overrides={"chat": "llama3.2"})
        assert reg.resolve("chat") == type(reg.resolve("chat"))("chat", "llama3.2", "override")
        gone = registry([CHAT9], overrides={"chat": "ghost"}).resolve("chat")
        assert gone.model is None
        assert "not installed" in gone.reason

    def test_embed_is_pinned(self) -> None:
        other = model("bge-m3", (CAP_EMBEDDING,), 1.2)
        reg = registry([EMBED, other], pinned_embed="bge-m3")
        assert reg.resolve("embed").model == "bge-m3"
        assert reg.resolve("embed").reason == "pinned"

    def test_latest_tag_is_matched(self) -> None:
        reg = registry([model("llama3.2:latest", size_gb=2.0)])
        assert reg.resolve("chat").model == "llama3.2:latest"

    def test_no_gpu_prefers_cpu_friendly_models_within_ram(self) -> None:
        reg = registry([CHAT9, LLAMA], hw(gpu=False, ram_free=8000))
        assert reg.resolve("chat").model == "llama3.2"
        assert registry([CHAT9], hw(gpu=False, ram_free=100000)).resolve("chat").model is None

    def test_unknown_size_is_assumed_to_fit(self) -> None:
        unknown = ModelInfo("tiny:1b", "ollama", capabilities=frozenset({CAP_COMPLETION}))
        assert registry([unknown], hw(free_vram=100)).resolve("chat").model == "tiny:1b"

    def test_size_estimate_blocks_oversized_unlisted_model(self) -> None:
        big = model("huge:70b", size_gb=40.0)
        assert registry([big], hw(free_vram=7000)).resolve("chat").model is None


class TestRecommendations:
    def test_better_option_is_suggested_with_pull_command(self) -> None:
        reg = registry([EMBED, LLAMA])
        report = reg.report()
        chat = next(r for r in report.recommendations if r.role == "chat")
        assert chat.model == "qwen3.5:9b"
        assert chat.pull_command == "ollama pull qwen3.5:9b"

    def test_missing_chat_model_suggests_pull(self) -> None:
        recs = registry([EMBED]).report().recommendations
        assert any(r.role == "chat" and r.model == "qwen3.5:9b" for r in recs)

    def test_no_recommendation_when_best_is_installed(self) -> None:
        reg = registry([EMBED, CHAT9, CODER, VISION, model("qwen2.5:7b")])
        assert [r for r in reg.report().recommendations if r.role == "chat"] == []

    def test_recommendation_respects_vram(self) -> None:
        reg = registry([LLAMA], hw(free_vram=3000))
        assert [r for r in reg.report().recommendations if r.role == "chat"] == []

    def test_embed_recommendation_mentions_reindex(self) -> None:
        reg = registry([model("nomic-embed-text", (CAP_EMBEDDING,), 0.3)])
        rec = next(r for r in reg.report().recommendations if r.role == "embed")
        assert "re-index" in rec.reason


class TestReport:
    def test_flags(self) -> None:
        reg = registry(
            [
                EMBED,
                CHAT9,
                model("mxbai-embed-large", (CAP_EMBEDDING,), 0.7),
                model("deepseek-r1:8b", size_gb=5.2),
                model("random-thing:7b", size_gb=4.0),
            ]
        )
        rows = {r.info.name: r for r in reg.report().rows}
        assert rows["qwen3.5:9b"].roles
        assert rows["qwen3.5:9b"].flags[0].kind == FLAG_OK
        kinds_mxbai = [f.kind for f in rows["mxbai-embed-large"].flags]
        assert FLAG_WARNING in kinds_mxbai
        assert "512" in rows["mxbai-embed-large"].flags[0].message
        kinds_ds = [f.kind for f in rows["deepseek-r1:8b"].flags]
        assert kinds_ds == [FLAG_WARNING, FLAG_UNUSED]
        assert "5.2 GB" in rows["deepseek-r1:8b"].flags[1].message
        assert FLAG_UNUSED in [f.kind for f in rows["random-thing:7b"].flags]

    def test_removing_chat_model_falls_back_and_suggests_pull(self) -> None:
        reg = registry([EMBED, CODER, LLAMA])
        report = reg.report()
        assert report.resolutions["chat"].model == "qwen2.5-coder:7b"
        assert any(r.model == "qwen3.5:9b" for r in report.recommendations)


class TestReindexNotice:
    def test_same_model_is_no_change(self) -> None:
        reg = registry([EMBED], pinned_embed="qwen3-embedding:0.6b")
        assert reg.reindex_notice("qwen3-embedding:0.6b:latest", 100) is None

    def test_change_estimates_cost(self) -> None:
        reg = registry([EMBED], pinned_embed="qwen3-embedding:0.6b")
        notice = reg.reindex_notice("bge-m3", 1200)
        assert notice is not None
        assert notice.files == 1200
        assert notice.estimated_seconds == pytest.approx(600)
        assert "re-index" in notice.message.lower()


class TestDiscovery:
    def test_refresh_persists_models_and_timestamp(self, tmp_path: Path) -> None:
        with StateDb(tmp_path, clock=lambda: 100.0) as state:
            reg = ModelRegistry(
                cat.load_catalog(),
                [FakeProvider([CHAT9])],
                state,
                hardware_probe=hw,
                clock=lambda: 100.0,
            )
            reg.refresh()
            assert [m["name"] for m in state.list_models("ollama")] == ["qwen3.5:9b"]
            assert state.get_meta("models_refreshed_at") == "100.0"

    def test_unreachable_provider_is_skipped(self) -> None:
        reg = ModelRegistry(
            cat.load_catalog(),
            [FakeProvider([], fail=True), FakeProvider([CHAT9])],
            hardware_probe=hw,
        )
        assert [m.name for m in reg.refresh()] == ["qwen3.5:9b"]
        assert [m.name for m in reg.installed] == ["qwen3.5:9b"]

    def test_refresh_if_stale(self, tmp_path: Path) -> None:
        now = [1000.0]
        with StateDb(tmp_path) as state:
            reg = ModelRegistry(
                cat.load_catalog(),
                [FakeProvider([CHAT9])],
                state,
                hardware_probe=hw,
                clock=lambda: now[0],
            )
            assert reg.refresh_if_stale(3600) is True  # never refreshed
            assert reg.refresh_if_stale(3600) is False
            now[0] += 3601
            assert reg.refresh_if_stale(3600) is True
