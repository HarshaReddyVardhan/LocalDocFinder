from pathlib import Path

import numpy as np
from tests.core.providers.fakes import FakeOllamaClient

from vector_embed.core import runtime
from vector_embed.core.doctypes.base import PROTOTYPE_TEXTS
from vector_embed.core.providers.ollama import OllamaProvider
from vector_embed.core.secrets import MemoryKeyStore
from vector_embed.core.settings import ImageSettings, ScopeSettings, Settings, StorageSettings
from vector_embed.core.store.sqlite import StateDb


def make_settings(tmp_path: Path, **kw: object) -> Settings:
    default = ScopeSettings()
    scope = ScopeSettings(blocked_dirs=default.blocked_dirs - {"appdata"})
    return Settings(scope=scope, storage=StorageSettings(data_dir=tmp_path / "data"), **kw)  # type: ignore[arg-type]


def make_provider(settings: Settings, client: FakeOllamaClient | None = None) -> OllamaProvider:
    return OllamaProvider(settings.embedding, client=client or FakeOllamaClient())


def test_scope_never_indexes_the_data_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    scope = runtime.build_scope(settings)
    inside = settings.storage.data_dir / "notes.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("# secret index", encoding="utf-8")
    assert not scope.is_valid_file(inside)


def test_projects_use_the_configured_roots(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    projects = runtime.build_projects(settings, runtime.build_scope(settings))
    assert [str(r) for r in projects.roots] == list(settings.scope.roots)


def test_open_store_probes_dim_only_when_the_model_changes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeOllamaClient()
    provider = make_provider(settings, client)
    with StateDb(settings.storage.data_dir) as state:
        store = runtime.open_store(settings, state, provider)
        assert store.dim == 4
        probes = len(client.calls)
        assert probes == 1
        runtime.open_store(settings, state, provider)
        assert len(client.calls) == probes  # model unchanged: dim came from the state db


def test_read_only_store_does_not_create_tables(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    with StateDb(settings.storage.data_dir) as state:
        store = runtime.open_read_only_store(settings, state)
        assert store.chunks is None


def test_prototypes_are_embedded_once_per_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeOllamaClient(
        embed_fn=lambda texts: [[1.0, 0.0, 0.0, float(len(t))] for t in texts]
    )
    provider = make_provider(settings, client)
    with StateDb(settings.storage.data_dir) as state:
        first = runtime.load_prototypes(state, provider)
        calls = len(client.calls)
        second = runtime.load_prototypes(state, provider)
        assert len(client.calls) == calls
        assert set(first) == set(PROTOTYPE_TEXTS)
        assert np.allclose(first["resume"], second["resume"])

        other = OllamaProvider(
            settings.embedding.model_copy(update={"model": "other"}), client=client
        )
        runtime.load_prototypes(state, other)
        assert len(client.calls) > calls  # a different model needs its own prototypes


def test_extractors_get_thumbs_dir_and_optional_captioner(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    provider = make_provider(settings)
    scope = runtime.build_scope(settings)
    plain = runtime.build_extractors(settings, scope, provider)
    assert plain.ctx.captioner is None
    assert plain.ctx.thumbs_dir == settings.storage.data_dir / runtime.THUMBS_DIRNAME

    captions = make_settings(tmp_path, images=ImageSettings(enable_captions=True))
    with_captions = runtime.build_extractors(captions, scope, provider)
    assert with_captions.ctx.captioner is not None
    assert runtime.build_extractors(captions, scope).ctx.captioner is None


def test_log_dir_is_under_the_data_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert runtime.log_dir(settings) == settings.storage.data_dir / "logs"


def test_provider_factory_uses_the_configured_host(tmp_path: Path) -> None:
    provider = runtime.build_provider(make_settings(tmp_path, ollama_host="http://example:9"))
    assert provider.embed_model == "qwen3-embedding:0.6b"


class TestCloudWiring:
    def cloud_settings(self, tmp_path: Path) -> Settings:
        from vector_embed.core.settings import CloudProviderSettings, CloudSettings

        cloud = CloudSettings(
            providers={
                "openrouter": CloudProviderSettings(
                    base_url="https://openrouter.ai/api/v1", models={"chat": "vendor/m"}
                )
            },
            active="openrouter",
            routing={"chat": "auto"},
        )
        return make_settings(tmp_path, cloud=cloud)

    def registry(self, settings: Settings, state: StateDb):  # type: ignore[no-untyped-def]
        return runtime.build_model_registry(settings, state, make_provider(settings))

    def test_nothing_can_leave_the_machine_by_default(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        with StateDb(settings.storage.data_dir) as state:
            ctx = runtime.build_cloud(
                settings, state, runtime.build_scope(settings), self.registry(settings, state)
            )
            assert ctx.provider is None
            assert not ctx.consent.granted
            assert ctx.router.model_for("chat") is None

    def test_a_stored_key_enables_the_cloud_provider_but_consent_is_still_needed(
        self, tmp_path: Path
    ) -> None:
        from vector_embed.core.secrets import MemoryKeyStore

        settings = self.cloud_settings(tmp_path)
        with StateDb(settings.storage.data_dir) as state:
            ctx = runtime.build_cloud(
                settings,
                state,
                runtime.build_scope(settings),
                self.registry(settings, state),
                MemoryKeyStore({"openrouter": "sk-test-1234567890abcdef"}),
            )
            assert ctx.provider is not None
            assert ctx.router.model_for("chat") == "vendor/m"
            assert not ctx.consent.granted

    def test_a_missing_or_unreadable_key_keeps_everything_local(self, tmp_path: Path) -> None:
        from vector_embed.core.secrets import KeyStoreError, MemoryKeyStore

        settings = self.cloud_settings(tmp_path)

        class Broken:
            def get(self, provider: str) -> str | None:
                raise KeyStoreError("vault locked")

        with StateDb(settings.storage.data_dir) as state:
            scope = runtime.build_scope(settings)
            registry = self.registry(settings, state)
            no_key = runtime.build_cloud(settings, state, scope, registry, MemoryKeyStore())
            broken = runtime.build_cloud(settings, state, scope, registry, Broken())  # type: ignore[arg-type]
            assert no_key.provider is None
            assert broken.provider is None

    def test_the_skill_context_routes_through_the_cloud_router(self, tmp_path: Path) -> None:
        from vector_embed.core.llm import LlmGateway
        from vector_embed.core.privacy.policy import PrivacyFilter
        from vector_embed.core.secrets import MemoryKeyStore

        settings = self.cloud_settings(tmp_path)
        with StateDb(settings.storage.data_dir) as state:
            ctx = runtime.build_skill_context(
                settings, state, keys=MemoryKeyStore({"openrouter": "sk-test-1234567890abcdef"})
            )
            gateway = ctx.extras["llm"]
            assert isinstance(gateway, LlmGateway)
            assert gateway.router is ctx.extras["cloud"].router
            assert isinstance(ctx.extras["privacy"], PrivacyFilter)


def test_mask_ids_locally_installs_the_local_filter(tmp_path: Path) -> None:
    from vector_embed.core.settings import PrivacySettings

    for enabled in (False, True):
        settings = make_settings(tmp_path, privacy=PrivacySettings(mask_ids_locally=enabled))
        with StateDb(settings.storage.data_dir) as state:
            ctx = runtime.build_skill_context(settings, state, keys=MemoryKeyStore())
            gateway = ctx.extras["llm"]
            assert (gateway.local_filter is not None) is enabled
