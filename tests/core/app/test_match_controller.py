import json
from pathlib import Path

import pytest
from tests.core.conftest import Chat, CloudRig, Env
from tests.core.match.test_pipeline import JD, faithful_model, resume, write

from localdoc_finder.app.match_controller import LOCAL, MatchController, format_tokens
from localdoc_finder.core.match.judge import MatchError
from localdoc_finder.core.match.recall import MatchCandidate
from localdoc_finder.core.skills.base import SkillContext


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


def test_match_runs_as_one_session_and_unloads_at_the_end(
    controller: MatchController, chat: Chat
) -> None:
    gateway = chat.gateway
    controller.start(JD)
    controller.checklist()
    assert gateway.session_active  # held across the steps: the model loads once
    controller.score()
    "".join(controller.verdict())
    assert gateway.session_active
    chat_calls = chat.chat_calls()
    assert chat_calls
    assert all(call["keep_alive"] == "10m" for call in chat_calls)  # never unloaded between calls
    controller.finish()
    assert not gateway.session_active
    assert chat.client.calls[-1][0] == "generate"  # the explicit unload
    assert chat.client.calls[-1][1]["keep_alive"] == 0
    controller.finish()  # a second finish is harmless


def test_the_route_is_worked_out_once_for_all_table_rows(
    controller: MatchController, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = controller.start(JD)
    calls: list[int] = []
    gateway = controller.pipeline.gateway
    original = gateway.cloud_destination
    monkeypatch.setattr(
        gateway, "cloud_destination", lambda role: calls.append(1) or original(role)
    )
    for _ in range(5):  # a redraw: every row, plus the footer, ask where it would go
        for candidate in run.candidates:
            controller.candidate_row(candidate)
        controller.footer()
    assert len(calls) <= 1
    controller.reset_context()  # a setting changed: look again
    controller.candidate_row(run.candidates[0])
    assert len(calls) == 2


def test_the_users_ticks_are_kept_for_later_runs(controller: MatchController) -> None:
    run = controller.start(JD)
    by_name = {c.name: c for c in run.candidates}
    defaults = {name: c.selected for name, c in by_name.items()}
    flipped = "Resume_b.txt"
    controller.choose(by_name[flipped], not defaults[flipped])
    again = {c.name: c.selected for c in controller.start(JD + " Also Go.").candidates}
    assert again[flipped] is not defaults[flipped]  # the user's choice survived a new JD
    assert again["Resume_a.txt"] is defaults["Resume_a.txt"]  # untouched: the default applies

    controller.choose_all(False)
    assert not any(c.selected for c in controller.start(JD).candidates)
    controller.top(1)
    assert sum(c.selected for c in controller.start(JD).candidates) == 1


def test_the_personal_details_box_overrides_the_setting_for_the_session(
    controller: MatchController, cloud: CloudRig
) -> None:
    assert not controller.redacts_personal  # nothing loaded yet: shown off until the first run
    controller.set_redact_personal(True)  # before any run: applied when the run starts
    assert not cloud.privacy.redacts_personal
    controller.start(JD)
    assert cloud.privacy.redacts_personal and controller.redacts_personal
    controller.set_redact_personal(False)  # with a run: applied at once
    assert not cloud.privacy.redacts_personal
