import numpy as np
import pytest
from tests.core.conftest import Env, Power

from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.retrieval import fts_terms, hybrid_candidates
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.store.lance import CHUNKS, DOCUMENTS

COLUMNS = ["path", "chunk_hash", "text"]


def index(env: Env, name: str, text: str) -> str:
    path = env.root / name
    path.write_text(text, encoding="utf-8")
    env.indexer.index_paths([str(path)])
    return str(path)


def candidates(ctx: SkillContext, text: str, **kw: object) -> list[str]:
    ctx.store.maintain()
    found = hybrid_candidates(
        ctx.store, ctx.embedder, ctx.power, ctx.settings.search,
        table=CHUNKS, text=text, columns=COLUMNS, **kw,
    )  # type: ignore[arg-type]  # fmt: skip
    return [c.row["path"] for c in found]


def test_terms_keep_identifiers_and_drop_punctuation() -> None:
    assert fts_terms("charge_card(order)!") == "charge_card order"


def test_vector_and_keyword_legs_are_fused(env: Env, skill_ctx: SkillContext) -> None:
    pay = index(env, "pay.txt", "retry failed payments with backoff\n")
    index(env, "other.txt", "completely unrelated gardening notes\n")
    assert candidates(skill_ctx, "retry failed payments")[0] == pay


def test_scores_are_descending_and_unique_per_chunk(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    skill_ctx.store.maintain()
    found = hybrid_candidates(
        skill_ctx.store, skill_ctx.embedder, skill_ctx.power, skill_ctx.settings.search,
        table=CHUNKS, text="alpha beta", columns=COLUMNS,
    )  # fmt: skip
    scores = [c.score for c in found]
    assert scores == sorted(scores, reverse=True)
    assert len({(c.row["path"], c.row["chunk_hash"]) for c in found}) == len(found)


def test_force_cpu_embeds_the_query_on_the_cpu(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    seen: list[bool] = []
    original = skill_ctx.embedder.embed

    def spy(texts: list[str], kind: str = "doc", cpu: bool = False) -> np.ndarray:
        seen.append(cpu)
        return original(texts, kind, cpu)  # type: ignore[arg-type]

    skill_ctx.embedder.embed = spy  # type: ignore[method-assign]
    candidates(skill_ctx, "alpha", force_cpu=True)
    candidates(skill_ctx, "alpha")
    assert seen == [True, False]


def test_battery_forces_cpu_too(env: Env, skill_ctx: SkillContext, power_state: Power) -> None:
    index(env, "a.txt", "alpha beta gamma\n")
    power_state.on_ac = False
    seen: list[bool] = []
    original = skill_ctx.embedder.embed
    skill_ctx.embedder.embed = lambda texts, kind="doc", cpu=False: (  # type: ignore[method-assign]
        seen.append(cpu) or original(texts, kind, cpu)  # type: ignore[arg-type]
    )
    candidates(skill_ctx, "alpha")
    assert seen == [True]


def test_keyword_only_when_the_model_server_is_down(
    env: Env, skill_ctx: SkillContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = index(env, "ident.txt", "the function charge_card handles billing\n")

    def down(*_a: object, **_k: object) -> None:
        raise ProviderError("down")

    monkeypatch.setattr(skill_ctx.embedder, "embed", down)
    assert candidates(skill_ctx, "charge_card") == [path]


def test_works_on_the_documents_table(env: Env, skill_ctx: SkillContext) -> None:
    index(env, "resume.txt", "python developer building payment systems\n" * 3)
    skill_ctx.store.maintain()
    found = hybrid_candidates(
        skill_ctx.store, skill_ctx.embedder, skill_ctx.power, skill_ctx.settings.search,
        table=DOCUMENTS, text="python payment developer", columns=["path", "doc_type"],
        unique_key=("path", "path"),
    )  # fmt: skip
    assert len(found) == 1
