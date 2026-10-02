import pytest

from vector_embed.core.models.catalog import ROLE_CHAT, ROLE_EMBED, load_catalog
from vector_embed.core.models.fit import budget_mb, fits
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.models.starter import pick_starter


def machine(vram_mb: int = 0, ram_mb: int = 32000) -> Hardware:
    gpu = vram_mb > 0
    return Hardware(
        gpu_name="GPU" if gpu else None,
        vram_total_mb=vram_mb,
        vram_free_mb=vram_mb // 2,  # free is deliberately lower: starter must use total
        ram_total_mb=ram_mb,
        ram_free_mb=ram_mb // 4,
        cpu_count=8,
        on_ac=True,
    )


@pytest.mark.parametrize(
    ("hardware", "embed", "chat"),
    [
        (machine(8192), "qwen3-embedding:0.6b", "qwen3.5:9b"),
        (machine(6144), "qwen3-embedding:0.6b", "qwen3:8b"),
        (machine(4096), "qwen3-embedding:0.6b", "llama3.2"),
        (machine(0, 16000), "qwen3-embedding:0.6b", "llama3.2"),
        (machine(0, 6000), "qwen3-embedding:0.6b", None),
        (machine(0, 4000), "nomic-embed-text", None),
        (machine(512), None, None),
    ],
    ids=["8gb", "6gb", "4gb", "cpu-16gb", "cpu-6gb", "cpu-4gb", "tiny-gpu"],
)
def test_pick_starter_by_hardware(hardware: Hardware, embed: str | None, chat: str | None) -> None:
    plan = pick_starter(load_catalog(), hardware)
    picked = {p.role: p.model for p in plan.picks}
    assert picked.get(ROLE_EMBED) == embed
    assert picked.get(ROLE_CHAT) == chat
    assert set(plan.missing_roles) == {
        role for role, model in ((ROLE_EMBED, embed), (ROLE_CHAT, chat)) if model is None
    }


def test_plan_totals_and_reasons_for_an_8gb_card() -> None:
    plan = pick_starter(load_catalog(), machine(8192))
    assert plan.models == ("qwen3-embedding:0.6b", "qwen3.5:9b")
    assert plan.total_download_mb == 640 + 6100
    chat = plan.pick_for(ROLE_CHAT)
    assert chat is not None
    assert "8192 MB of VRAM" in chat.reason
    assert plan.pick_for("caption") is None


def test_cpu_reason_mentions_ram() -> None:
    plan = pick_starter(load_catalog(), machine(0, 16000))
    pick = plan.pick_for(ROLE_CHAT)
    assert pick is not None
    assert "CPU-only" in pick.reason


def test_downgrade_is_next_smaller_fitting_model() -> None:
    plan = pick_starter(load_catalog(), machine(8192))
    chat = plan.pick_for(ROLE_CHAT)
    embed = plan.pick_for(ROLE_EMBED)
    assert chat is not None
    assert embed is not None
    assert chat.downgrade == "qwen3:8b"
    assert embed.downgrade == "embeddinggemma"
    smallest = pick_starter(load_catalog(), machine(4096)).pick_for(ROLE_CHAT)
    assert smallest is not None
    assert smallest.downgrade is None


def test_shared_model_is_counted_once() -> None:
    plan = pick_starter(load_catalog(), machine(8192), roles=(ROLE_CHAT, "match_scorer"))
    assert plan.models == ("qwen3.5:9b",)
    assert plan.total_download_mb == 6100


def test_models_missing_from_the_catalog_table_are_skipped() -> None:
    from vector_embed.core.models.catalog import parse_catalog

    catalog = parse_catalog(
        '[roles]\nchat = ["ghost", "tiny"]\n[models.tiny]\nvram_mb = 100\ndownload_mb = 5\n'
    )
    plan = pick_starter(catalog, machine(8192), roles=(ROLE_CHAT,))
    assert plan.models == ("tiny",)


def test_fit_budget_uses_total_or_free() -> None:
    gpu = machine(8192)
    assert budget_mb(gpu) == 4096
    assert budget_mb(gpu, total=True) == 8192
    cpu = machine(0, 16000)
    assert budget_mb(cpu) == 2000
    assert budget_mb(cpu, total=True) == 8000


def test_fits_treats_unknown_size_as_fitting_and_checks_cpu_rules() -> None:
    cat = load_catalog()
    cpu = machine(0, 16000)
    assert fits(None, None, cpu, 100)
    assert not fits(cat.entry("qwen3.5:9b"), 100, cpu, 10_000)  # not cpu_ok
    assert not fits(cat.entry("llama3.2"), 2300, machine(0, 4000), 2000)  # below min_ram_mb
