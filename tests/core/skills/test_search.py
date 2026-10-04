import os
from pathlib import Path

import pytest
from tests.core.conftest import Env, Power

from localdoc_finder.core.providers.base import ProviderError
from localdoc_finder.core.skills import search as search_mod
from localdoc_finder.core.skills.base import SKILLS, SkillContext, create_skill, load_skills
from localdoc_finder.core.skills.search import (
    SearchDisabledError,
    SearchInput,
    SearchSkill,
    parse_query,
)
from localdoc_finder.core.store.lance import IndexRebuildingError

PAYMENTS = (
    "def retry_failed_payments(order):\n"
    "    # retry charging the card after a payment failure\n"
    "    return charge_card(order)\n"
)
GREET = "def greet(name):\n    return 'hello ' + name\n"
NOTES = "# Meeting\n\nWe decided to retry failed payments with exponential backoff.\n"


def write(env: Env, rel: str, content: str) -> str:
    path = env.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture
def indexed(env: Env) -> dict[str, str]:
    write(env, "billing/pyproject.toml", "[project]\n")
    write(env, "web/pyproject.toml", "[project]\n")
    paths = {
        "payments": write(env, "billing/payments.py", PAYMENTS),
        "greet": write(env, "web/greet.py", GREET),
        "notes": write(env, "billing/meeting.md", NOTES),
    }
    env.indexer.index_paths(list(paths.values()))
    env.store.maintain()
    return paths


@pytest.fixture
def skill(skill_ctx: SkillContext) -> SearchSkill:
    return SearchSkill(skill_ctx)


class TestParseQuery:
    def test_plain_text_has_no_filters(self) -> None:
        assert parse_query("retry payments") == search_mod.ParsedQuery("retry payments", "")

    def test_filters_are_extracted_and_combined(self) -> None:
        parsed = parse_query("retry type:code proj:Billing ext:py")
        assert parsed.text == "retry"
        assert "kind IN ('code','outline')" in parsed.where
        assert "lower(project) = 'billing'" in parsed.where
        assert "ext = '.py'" in parsed.where
        assert parsed.where.count(" AND ") == 2

    def test_type_aliases_and_extensions(self) -> None:
        assert "kind = 'image'" in parse_query("type:img").where
        assert "source = 'claude-plan'" in parse_query("type:plan").where
        assert "ext = '.pdf'" in parse_query("type:pdf").where
        assert parse_query("type:code,pdf").where.count(" OR ") == 1
        assert parse_query("type:,").where == ""

    def test_in_and_dates(self) -> None:
        parsed = parse_query(r'x in:"D:\Projects" after:2026-01 before:2026-06-15 after:nonsense')
        assert "starts_with(lower(path)," in parsed.where
        assert "mtime >=" in parsed.where
        assert "mtime <" in parsed.where
        assert parsed.where.count("mtime") == 2  # the unparseable date is dropped

    def test_quotes_are_escaped(self) -> None:
        assert "o''brien" in parse_query("proj:o'brien").where

    @pytest.mark.parametrize("value", ["2026-01-05", "2026-01", "2026"])
    def test_date_formats(self, value: str) -> None:
        assert search_mod._timestamp(value) is not None


class TestSearch:
    def test_semantic_and_keyword_hits_rank_the_relevant_file_first(
        self, skill: SearchSkill, indexed: dict[str, str]
    ) -> None:
        results = skill.search("retry failed payments")
        names = [Path(r.path).name for r in results]
        assert names[0] in {"payments.py", "meeting.md"}
        assert "greet.py" not in names[:2]
        assert results[0].location

    def test_exact_identifier_is_found_by_the_keyword_leg(
        self, skill: SearchSkill, indexed: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(*_a: object, **_k: object) -> None:
            raise ProviderError("ollama down")

        monkeypatch.setattr(skill.ctx.embedder, "embed", down)
        results = skill.search("charge_card")
        assert [Path(r.path).name for r in results] == ["payments.py"]

    def test_filters_restrict_results(self, skill: SearchSkill, indexed: dict[str, str]) -> None:
        only_md = skill.search("retry payments ext:md")
        assert {Path(r.path).suffix for r in only_md} == {".md"}
        only_web = skill.search("hello proj:web")
        assert {r.project for r in only_web} == {"web"}
        inside = skill.search(f'hello in:"{Path(indexed["greet"]).parent}"')
        assert [Path(r.path).name for r in inside] == ["greet.py"]

    def test_filter_only_query_lists_newest_first(
        self, skill: SearchSkill, indexed: dict[str, str], env: Env
    ) -> None:
        os.utime(indexed["greet"], (1, 2_000_000_000))
        env.indexer.index_paths([indexed["greet"]], force=True)
        env_files = skill.search("ext:py")
        assert env_files[0].path == indexed["greet"]

    def test_filename_match_is_boosted(self, skill: SearchSkill, indexed: dict[str, str]) -> None:
        plain = skill.ctx.settings.search.filename_boost
        assert plain > 1
        boosted = {r.path: r.score for r in skill.search("greet")}
        assert boosted[indexed["greet"]] > 0

    def test_file_named_in_the_query_ranks_first(self, env: Env, skill: SearchSkill) -> None:
        # The content never mentions the name, and every other file shares the .txt extension.
        paths = [write(env, "docs/References.txt", "alpha\n")] + [
            write(env, f"docs/other{n}.txt", f"references txt references list {n}\n")
            for n in range(6)
        ]
        env.indexer.index_paths(paths)
        env.store.maintain()
        # The named file falls outside both legs' top-N.
        settings = skill.ctx.settings
        narrow = settings.search.model_copy(update={"candidates": 2})
        skill.ctx.settings = settings.model_copy(update={"search": narrow})
        for query in ("references.txt", "references"):
            assert Path(skill.search(query)[0].path).name == "References.txt"

    def test_extension_alone_is_not_a_filename_match(self, skill: SearchSkill) -> None:
        cfg = skill.ctx.settings.search
        assert search_mod._name_boost("D:/a/notes.txt", "txt", cfg) == 1.0
        assert (
            search_mod._name_boost("D:/a/notes.txt", "notes.txt", cfg) == cfg.filename_exact_boost
        )
        assert search_mod._name_boost("D:/a/notes.txt", "notes", cfg) == cfg.filename_exact_boost
        half = search_mod._name_boost("D:/a/notes.txt", "notes budget", cfg)
        assert 1.0 < half < cfg.filename_boost

    def test_current_project_is_boosted(self, skill: SearchSkill, indexed: dict[str, str]) -> None:
        base = {r.path: r.score for r in skill.search("retry payments")}
        boosted = {
            r.path: r.score for r in skill.search("retry payments", current_project="Billing")
        }
        assert boosted[indexed["payments"]] > base[indexed["payments"]]
        assert (
            boosted[indexed["greet"]] == pytest.approx(base[indexed["greet"]])
            if indexed["greet"] in boosted
            else True
        )

    def test_limit_and_extra_hits(self, skill: SearchSkill, indexed: dict[str, str]) -> None:
        assert len(skill.search("retry payments hello", limit=1)) == 1
        multi = [
            r for r in skill.search("retry payments card charge") if r.path == indexed["payments"]
        ]
        assert multi[0].extra_hits >= 1

    def test_empty_index_returns_nothing(self, skill: SearchSkill) -> None:
        assert skill.search("anything") == []

    def test_search_stays_off_the_gpu_while_another_process_chats(
        self, skill: SearchSkill, indexed: dict[str, str]
    ) -> None:
        calls: list[bool] = []
        original = skill.ctx.embedder.embed

        def spy(texts: list[str], kind: str = "doc", cpu: bool = False):  # type: ignore[no-untyped-def]
            calls.append(cpu)
            return original(texts, kind, cpu)  # type: ignore[arg-type]

        skill.ctx.embedder.embed = spy  # type: ignore[method-assign]
        skill.search("retry payments")
        assert calls == [False]  # nobody chatting: the GPU is fine
        skill.ctx.state.acquire_lock("chat", "the-app-in-another-process", 60)
        calls.clear()
        assert skill.search("retry payments")
        assert calls == [True]

    def test_battery_policy(
        self, skill: SearchSkill, indexed: dict[str, str], power_state: Power
    ) -> None:
        power_state.on_ac = False
        calls: list[bool] = []
        original = skill.ctx.embedder.embed

        def spy(texts: list[str], kind: str = "doc", cpu: bool = False):  # type: ignore[no-untyped-def]
            calls.append(cpu)
            return original(texts, kind, cpu)  # type: ignore[arg-type]

        skill.ctx.embedder.embed = spy  # type: ignore[method-assign]
        assert skill.search("retry payments")
        assert calls == [True]  # query embedded on the CPU while unplugged
        disabled = skill.ctx.settings.model_copy(
            update={
                "power": skill.ctx.settings.power.model_copy(update={"search_on_battery": False})
            }
        )
        skill.ctx.power._settings = disabled.power
        with pytest.raises(SearchDisabledError):
            skill.search("retry payments")

    def test_search_refuses_to_mix_vectors_from_another_embedder(
        self, skill: SearchSkill, indexed: dict[str, str]
    ) -> None:
        # The user switched embedders; the worker has not rebuilt the index yet.
        skill.ctx.store.model_id = "another-model"
        with pytest.raises(IndexRebuildingError, match="being rebuilt"):
            skill.search("retry payments")


class TestSkillInterface:
    def test_run_render_and_warm(self, skill: SearchSkill, indexed: dict[str, str]) -> None:
        results = skill.run(SearchInput(query="retry failed payments", limit=2))
        text = skill.render(results)
        assert "payments" in text.lower()
        assert skill.render([]) == "no results"
        skill.warm()
        assert any(kind == "query" for _, kind in skill.ctx.embedder.calls)  # type: ignore[attr-defined]

    def test_warm_ignores_provider_errors(
        self, skill: SearchSkill, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(*_a: object, **_k: object) -> None:
            raise ProviderError("down")

        monkeypatch.setattr(skill.ctx.embedder, "embed", down)
        skill.warm()

    def test_registry_discovers_search(self, skill_ctx: SkillContext) -> None:
        assert SearchSkill in load_skills()
        assert "search" in SKILLS
        assert isinstance(create_skill("search", skill_ctx), SearchSkill)
        assert SearchSkill.cli_positional == "query"
        assert "query" in SearchSkill.Input.model_json_schema()["properties"]

    def test_a_new_skill_is_one_decorated_class(self, skill_ctx: SkillContext) -> None:
        from localdoc_finder.core.skills.base import Skill, SkillInput, register_skill

        @register_skill("hello")
        class Hello(Skill):
            name = "hello"
            title = "Hello"
            description = "demo"

            def run(self, params: SkillInput) -> object:
                return "hi"

        try:
            assert create_skill("hello", skill_ctx).run(SkillInput()) == "hi"
            assert create_skill("hello", skill_ctx).render("x") == "x"
        finally:
            SKILLS.remove("hello")
