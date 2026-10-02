import json
from pathlib import Path

import pytest
from tests.core.conftest import Chat, Env
from tests.core.match.test_pipeline import JD, faithful_model, resume, write

from vector_embed.app.match_controller import LOCAL, MatchController, format_tokens
from vector_embed.core.match.judge import MatchError
from vector_embed.core.match.recall import MatchCandidate
from vector_embed.core.skills.base import SkillContext


@pytest.fixture
def controller(env: Env, chat: Chat, skill_ctx: SkillContext) -> MatchController:
    chat.client.chat_json_fn = faithful_model
    paths = [
        write(env, "Resume_a.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes", "AWS")),
        write(env, "Resume_b.txt", resume("Sam Roe", "Python")),
    ]
    env.indexer.index_paths(paths)
    env.store.maintain()
    return MatchController(lambda: skill_ctx)


def test_steps_run_in_order_and_expose_a_live_footer(controller: MatchController) -> None:
    run = controller.start(JD)
    assert {c.name for c in run.candidates} == {"Resume_a.txt", "Resume_b.txt"}
    assert controller.footer().startswith("JD + ")
    assert controller.footer().endswith("→ " + LOCAL)
    checklist = controller.checklist()
    assert [r.text for r in checklist] == ["Python", "PostgreSQL", "Kubernetes", "AWS"]
    messages: list[str] = []
    scores = controller.score(messages.append)
    assert len(scores) == len(messages) == sum(c.selected for c in run.candidates)
    assert "".join(controller.verdict())  # streams from the scripted model


def test_footer_changes_with_the_selection_and_mentions_locked_files(
    controller: MatchController,
) -> None:
    run = controller.start(JD)
    for candidate in run.candidates:
        candidate.selected = True
    both = controller.footer()
    assert "2 documents" in both
    run.candidates[0].selected = False
    assert "JD + 1 document ≈" in controller.footer()
    run.candidates[1].locked = True
    assert "🔒 1 scored locally" in controller.footer()
    run.candidates[1].selected = False
    assert controller.footer() == "nothing selected"


def test_destination_labels_come_from_the_injected_function(
    env: Env, chat: Chat, skill_ctx: SkillContext
) -> None:
    write(env, "Resume_a.txt", resume("Jane", "Python"))
    env.indexer.index_paths([str(env.root / "Resume_a.txt")])
    env.store.maintain()
    cloudy = MatchController(lambda: skill_ctx, destination=lambda c: "☁ openrouter")
    run = cloudy.start(JD)
    run.candidates[0].selected = True
    assert cloudy.footer().endswith("→ ☁ openrouter")
    assert cloudy.candidate_row(run.candidates[0])[-1] == "☁ openrouter"


def test_checklist_edits_are_kept(controller: MatchController) -> None:
    controller.start(JD)
    edited = controller.checklist()
    edited[0].enabled = False
    controller.set_checklist(edited)
    assert controller.run is not None
    assert controller.run.requirements[0].enabled is False
    assert controller.checklist()[0].enabled is False  # not regenerated


def test_top_and_add_file(controller: MatchController, env: Env) -> None:
    run = controller.start(JD)
    controller.top(1)
    assert sum(c.selected for c in run.candidates) == 1
    extra = write(env, "elsewhere/Resume_extra.txt", resume("Ann Lee", "Python", "AWS"))
    added = controller.add_file(extra)
    assert added.selected
    assert added in run.candidates
    controller.add_file(extra)  # adding the same file again does not duplicate it
    assert [c.path for c in run.candidates].count(added.path) == 1


def test_chat_state_pins_the_top_documents_and_carries_the_scores(
    controller: MatchController,
) -> None:
    run = controller.start(JD)
    for candidate in run.candidates:
        candidate.selected = True
    controller.checklist()
    controller.score()
    state = controller.chat_state(top=1)
    assert len(state.pinned) == 1
    assert "JOB DESCRIPTION" in state.scratch
    assert json.loads(state.scratch.split("\n", 6)[-1].split("\n")[-1])["summary"] == "Reasoned."


def test_calls_before_start_are_a_clear_error(controller: MatchController) -> None:
    with pytest.raises(MatchError, match="paste a job description"):
        controller.footer()
    with pytest.raises(MatchError):
        controller.checklist()


def test_candidate_rows(controller: MatchController, tmp_path: Path) -> None:
    newest = MatchCandidate(
        str(tmp_path / "Resume.pdf"), "t", "resume", 1_700_000_000, 0.8765, 2500, versions=3
    )
    older = MatchCandidate(
        str(tmp_path / "Resume_v1.pdf"),
        "t",
        "resume",
        1_600_000_000,
        0.5,
        900,
        versions=3,
        is_latest=False,
    )
    single = MatchCandidate(
        str(tmp_path / "Other.pdf"), "t", "resume", 1_700_000_000, 0.4, 100, locked=True
    )
    assert controller.candidate_row(newest)[3] == "3 versions, newest"
    assert controller.candidate_row(newest)[4:6] == ["0.88", "2.5k"]
    assert controller.candidate_row(older)[3] == "older version"
    row = controller.candidate_row(single)
    assert row[3] == ""
    assert row[-1] == "🔒 " + LOCAL
    assert row[0] == "Other.pdf"


def test_token_formatting() -> None:
    assert format_tokens(999) == "999"
    assert format_tokens(1500) == "1.5k"
