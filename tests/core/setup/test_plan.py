import pytest

from vector_embed.core.features import FEATURES
from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_EMBED, load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.setup.plan import (
    DISK_HEADROOM_MB,
    SetupChoices,
    SetupPlanError,
    has_enough_disk,
    plan_setup,
)

CATALOG = load_catalog()
GPU8 = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)
GPU4 = Hardware("GTX 1650", 4096, 3500, 16000, 8000, 8, True)
EVERYTHING = SetupChoices(features=FEATURES)


def test_search_only_plans_just_the_embedder() -> None:
    plan = plan_setup(CATALOG, GPU8)
    assert [(m.role, m.model) for m in plan.models] == [(ROLE_EMBED, "qwen3-embedding:0.6b")]
    assert plan.download_mb([]) == 640


def test_unknown_feature_is_rejected() -> None:
    with pytest.raises(SetupPlanError, match="bogus"):
        plan_setup(CATALOG, GPU8, SetupChoices(features=("bogus",)))


def test_auto_pick_on_an_8gb_card() -> None:
    plan = plan_setup(CATALOG, GPU8, EVERYTHING)
    assert [(m.role, m.model) for m in plan.models] == [
        (ROLE_EMBED, "qwen3-embedding:0.6b"),
        (ROLE_CHAT, "qwen3.5:9b"),
    ]
    assert plan.warnings == ()
    assert plan.download_mb([]) == 640 + 6100


def test_installed_models_are_not_downloaded_again() -> None:
    plan = plan_setup(CATALOG, GPU8)
    assert plan.to_download(["qwen3.5:9b"]) == ("qwen3-embedding:0.6b",)
    assert plan.to_download(["qwen3-embedding:0.6b:latest", "qwen3.5:9b"]) == ()
    assert plan.download_mb(["qwen3.5:9b"]) == 640


def test_explicit_choices_win_and_warn_when_they_do_not_fit() -> None:
    plan = plan_setup(CATALOG, GPU4, SetupChoices(chat="qwen3.5:9b", embed="nomic-embed-text"))
    chat = plan.model_for(ROLE_CHAT)
    assert chat is not None
    assert chat.model == "qwen3.5:9b"
    assert not chat.fits
    assert chat.reason == "chosen by you"
    assert any("qwen3.5:9b may not fit" in w for w in plan.warnings)
    assert plan.model_for(ROLE_EMBED) is not None


def test_a_model_outside_the_catalog_is_allowed_with_unknown_size() -> None:
    plan = plan_setup(CATALOG, GPU8, SetupChoices(chat="my/custom:7b"))
    chat = plan.model_for(ROLE_CHAT)
    assert chat is not None
    assert (chat.download_mb, chat.fits) == (0, True)
    assert "size unknown" in chat.reason


def test_extras_add_models_and_dedupe_shared_downloads() -> None:
    plan = plan_setup(
        CATALOG, GPU8, SetupChoices(extras=("match_scorer", "caption"), features=FEATURES)
    )
    names = [m.model for m in plan.models]
    assert names == ["qwen3-embedding:0.6b", "qwen3.5:9b", "qwen3.5:9b", "qwen2.5vl:3b"]
    assert plan.to_download([]) == ("qwen3-embedding:0.6b", "qwen3.5:9b", "qwen2.5vl:3b")
    assert plan.download_mb([]) == 640 + 6100 + 3200


def test_unknown_extra_role_is_rejected() -> None:
    with pytest.raises(SetupPlanError, match="bogus"):
        plan_setup(CATALOG, GPU8, SetupChoices(extras=("bogus",)))
    with pytest.raises(SetupPlanError, match="chat"):
        plan_setup(CATALOG, GPU8, SetupChoices(extras=("chat",)))


def test_locked_embedder_is_kept_with_a_warning() -> None:
    plan = plan_setup(CATALOG, GPU8, SetupChoices(embed="bge-m3"), locked_embed="embeddinggemma")
    embed = plan.model_for(ROLE_EMBED)
    assert embed is not None
    assert embed.model == "embeddinggemma"
    assert any("index already exists" in w for w in plan.warnings)


def test_missing_roles_become_warnings() -> None:
    tiny = Hardware(None, 0, 0, 3000, 1500, 2, True)
    plan = plan_setup(CATALOG, tiny, EVERYTHING)
    assert plan.models == ()
    assert "no chat model fits this machine" in plan.warnings


def test_disk_check_keeps_headroom() -> None:
    plan = plan_setup(CATALOG, GPU8)
    need = plan.download_mb([]) + DISK_HEADROOM_MB
    assert has_enough_disk(plan, [], need)
    assert not has_enough_disk(plan, [], need - 1)
