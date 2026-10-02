import time
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant
from tests.core.conftest import Chat, CloudRig, Env
from tests.core.match.test_pipeline import JD, faithful_model, resume, write

from vector_embed.app.assistant import ChatState, CloudPreview
from vector_embed.app.controller import Launcher
from vector_embed.app.match_controller import MatchController
from vector_embed.app.match_panel import (
    PAGE_CANDIDATES,
    PAGE_CHECKLIST,
    PAGE_RESULTS,
    MatchPanel,
)
from vector_embed.app.window import Mode, SearchWindow
from vector_embed.core.skills.base import SkillContext


def wait_for(qapp: QApplication, condition: Callable[[], bool], timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def controller(env: Env, chat: Chat, skill_ctx: SkillContext) -> MatchController:
    chat.client.chat_json_fn = faithful_model
    chat.client.chat_reply = ["Jane is the best fit."]
    paths = [
        write(env, "Resume_a.txt", resume("Jane Doe", "Python", "PostgreSQL", "Kubernetes", "AWS")),
        write(env, "Resume_b.txt", resume("Sam Roe", "Python")),
        write(env, "Resume_c.txt", resume("Joe Bloggs", "Java", "Spring")),
    ]
    env.indexer.index_paths(paths)
    env.store.maintain()
    return MatchController(lambda: skill_ctx)


@pytest.fixture
def panel(qapp: QApplication, controller: MatchController, env: Env) -> MatchPanel:
    extra = write(env, "elsewhere/Resume_extra.txt", resume("Ann Lee", "Python", "AWS"))
    return MatchPanel(controller, pick_file=lambda: extra)


def recall(qapp: QApplication, panel: MatchPanel) -> None:
    panel.begin(JD)
    wait_for(qapp, lambda: panel.candidates.rowCount() > 0)


class TestCandidates:
    def test_recall_fills_the_table_with_ticks_and_a_footer(
        self, qapp: QApplication, panel: MatchPanel
    ) -> None:
        recall(qapp, panel)
        assert panel.candidates.rowCount() == 3
        names = {panel.candidates.item(r, 1).text() for r in range(3)}
        assert names == {"Resume_a.txt", "Resume_b.txt", "Resume_c.txt"}
        assert panel.candidates.item(0, 0).checkState() in (
            Qt.CheckState.Checked,
            Qt.CheckState.Unchecked,
        )
        assert panel.footer.text().startswith(("JD + ", "nothing selected"))
        assert panel.pages.currentIndex() == PAGE_CANDIDATES

    def test_unticking_a_row_updates_the_run_and_the_footer(
        self, qapp: QApplication, panel: MatchPanel, controller: MatchController
    ) -> None:
        recall(qapp, panel)
        panel.buttons["all"].click()
        assert "3 documents" in panel.footer.text()
        panel.candidates.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        assert controller.run is not None
        assert sum(c.selected for c in controller.run.candidates) == 2
        assert "2 documents" in panel.footer.text()

    def test_select_helpers(
        self, qapp: QApplication, panel: MatchPanel, controller: MatchController
    ) -> None:
        recall(qapp, panel)
        panel.buttons["none"].click()
        assert panel.footer.text() == "nothing selected"
        panel.buttons["top3"].click()
        assert controller.run is not None
        assert sum(c.selected for c in controller.run.candidates) == 3

    def test_add_file_brings_in_a_document_recall_missed(
        self, qapp: QApplication, panel: MatchPanel, controller: MatchController
    ) -> None:
        recall(qapp, panel)
        panel.buttons["add"].click()
        assert panel.candidates.rowCount() == 4
        assert any(panel.candidates.item(r, 1).text() == "Resume_extra.txt" for r in range(4))

    def test_add_file_cancel_and_errors(
        self, qapp: QApplication, controller: MatchController, env: Env
    ) -> None:
        cancelled = MatchPanel(controller, pick_file=lambda: None)
        cancelled.add_file()  # no run yet and no file: nothing happens
        secret = write(env, ".env", "TOKEN=1")
        failing = MatchPanel(controller, pick_file=lambda: secret)
        failing.begin(JD)
        wait_for(qapp, lambda: failing.candidates.rowCount() > 0)
        failing.add_file()
        assert "secret" in failing.footer.text()

    def test_no_candidates_says_so(self, qapp: QApplication, panel: MatchPanel) -> None:
        panel.doc_type.setCurrentText("invoice")
        panel.begin(JD)
        wait_for(qapp, lambda: "no matching documents" in panel.footer.text())
        assert panel.candidates.rowCount() == 0

    def test_buttons_are_disabled_until_there_is_a_run(self, panel: MatchPanel) -> None:
        assert not panel.buttons["score"].isEnabled()
        assert not panel.buttons["checklist"].isEnabled()
        panel._tick_all(True)  # harmless without a run
        panel._tick_top3()
        panel.request_score()


class TestChecklistAndScoring:
    def test_checklist_page_allows_untick_and_reweight(
        self, qapp: QApplication, panel: MatchPanel, controller: MatchController
    ) -> None:
        recall(qapp, panel)
        panel.buttons["checklist"].click()
        wait_for(qapp, lambda: panel.checklist.rowCount() == 4)
        assert panel.pages.currentIndex() == PAGE_CHECKLIST
        assert panel.checklist.item(0, 1).text() == "Python"
        panel.checklist.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
        panel.checklist.item(1, 3).setText("3")
        panel.checklist.item(2, 3).setText("not a number")
        assert controller.run is not None
        reqs = controller.run.requirements
        assert reqs[0].enabled is False
        assert reqs[1].weight == 3.0
        assert panel.checklist.item(2, 3).text() == "1"  # bad input reverts
        panel.checklist.item(3, 3).setText("0")
        assert reqs[3].weight == 0.1  # clamped above zero
        panel.buttons["back"].click()
        assert panel.pages.currentIndex() == PAGE_CANDIDATES

    def test_score_ranks_results_and_streams_the_verdict(
        self, qapp: QApplication, panel: MatchPanel
    ) -> None:
        recall(qapp, panel)
        panel.buttons["all"].click()
        panel.buttons["score"].click()
        wait_for(qapp, lambda: panel.pages.currentIndex() == PAGE_RESULTS)
        assert panel.results.rowCount() == 3
        assert panel.results.item(0, 2).text() == "Resume_a.txt"
        assert panel.results.item(0, 1).text() == "100"
        assert "must-haves met" in panel.results.item(0, 3).text()
        assert panel.results.item(2, 1).text() == "0"
        wait_for(qapp, lambda: "best fit" in panel.verdict.toPlainText())
        assert "Jane is the best fit." in panel.verdict.toPlainText()
        assert panel.buttons["chat"].isVisible() or panel.buttons["chat"].isEnabled()

    def test_scoring_failures_are_reported(
        self, qapp: QApplication, panel: MatchPanel, chat: Chat
    ) -> None:
        recall(qapp, panel)
        panel.buttons["all"].click()
        chat.client.chat_json_fn = lambda _kw: '{"requirements": []}'
        panel.buttons["score"].click()
        wait_for(qapp, lambda: "score failed" in panel.footer.text())
        assert "no requirements" in panel.footer.text()

    def test_chat_hand_off_carries_pinned_documents_and_scores(
        self, qapp: QApplication, panel: MatchPanel
    ) -> None:
        recall(qapp, panel)
        panel.buttons["all"].click()
        panel.buttons["score"].click()
        wait_for(qapp, lambda: panel.pages.currentIndex() == PAGE_RESULTS)
        states: list[ChatState] = []
        panel.chat_requested.connect(states.append)
        panel.buttons["chat"].click()
        (state,) = states
        assert len(state.pinned) == 3
        assert "SCORING RESULTS" in state.scratch

    def test_chat_before_scoring_reports_instead_of_crashing(
        self, qapp: QApplication, panel: MatchPanel
    ) -> None:
        recall(qapp, panel)
        panel.request_chat()
        assert panel.footer.text()  # a message, not an exception

    def test_reset_clears_the_previous_match(self, qapp: QApplication, panel: MatchPanel) -> None:
        recall(qapp, panel)
        panel.reset()
        assert panel.candidates.rowCount() == 0
        assert panel.verdict.toPlainText() == ""


class TestInsideTheWindow:
    @pytest.fixture
    def window(
        self, qapp: QApplication, controller: MatchController, tmp_path: Path, env: Env
    ) -> tuple[SearchWindow, FakeAssistant]:
        assistant = FakeAssistant()
        window = SearchWindow(
            FakeService(),  # type: ignore[arg-type]
            Launcher(),
            tmp_path,
            assistant,  # type: ignore[arg-type]
            matcher=controller,
        )
        return window, assistant

    def test_tab_cycles_through_match_and_back(
        self, window: tuple[SearchWindow, FakeAssistant]
    ) -> None:
        win, _ = window
        win.show()
        modes = []
        for _ in range(4):
            QTest.keyClick(win.input, Qt.Key.Key_Tab)
            modes.append(win.mode)
        assert modes == [Mode.ASK, Mode.CHAT, Mode.MATCH, Mode.SEARCH]
        win.set_mode(Mode.MATCH)
        assert win.body.currentIndex() == 1
        win.set_mode(Mode.SEARCH)
        assert win.body.currentIndex() == 0

    def test_match_mode_is_unavailable_without_a_matcher(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        win = SearchWindow(FakeService(), Launcher(), tmp_path, FakeAssistant())  # type: ignore[arg-type]
        assert Mode.MATCH not in win.available_modes()
        win.set_mode(Mode.MATCH)
        assert win.mode is Mode.SEARCH

    def test_paste_and_enter_runs_recall_then_the_whole_flow_to_chat(
        self, qapp: QApplication, window: tuple[SearchWindow, FakeAssistant]
    ) -> None:
        win, assistant = window
        win.set_mode(Mode.MATCH)
        QGuiApplication.clipboard().setText(JD)
        QTest.keyClick(win.input, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
        assert "pasted text attached" in win.status.text()
        QTest.keyClick(win.input, Qt.Key.Key_Return)
        assert win.panel is not None
        wait_for(qapp, lambda: win.panel.candidates.rowCount() > 0)  # type: ignore[union-attr]
        win.panel.buttons["all"].click()
        win.panel.buttons["score"].click()
        wait_for(qapp, lambda: win.panel.pages.currentIndex() == PAGE_RESULTS)  # type: ignore[union-attr]
        win.panel.buttons["chat"].click()
        assert win.mode is Mode.CHAT
        assert len(win._chat.pinned) == 3
        assert "JOB DESCRIPTION" in win._chat.scratch
        wait_for(qapp, lambda: ("begin", None) in assistant.calls)

    def test_enter_without_text_asks_for_it_and_typed_text_works(
        self, qapp: QApplication, window: tuple[SearchWindow, FakeAssistant]
    ) -> None:
        win, _ = window
        win.set_mode(Mode.MATCH)
        win.start_match()
        assert "paste the text" in win.status.text()
        win.input.setText(JD)
        win.start_match()
        assert win.panel is not None
        wait_for(qapp, lambda: win.panel.candidates.rowCount() > 0)  # type: ignore[union-attr]

    def test_paste_outside_match_or_with_an_empty_clipboard_is_ignored(
        self, window: tuple[SearchWindow, FakeAssistant]
    ) -> None:
        win, _ = window
        QGuiApplication.clipboard().setText(JD)
        assert win.paste_job_description() is False  # not in Match mode
        win.set_mode(Mode.MATCH)
        QGuiApplication.clipboard().setText("   ")
        assert win.paste_job_description() is False


class TestCloudConsent:
    def test_declining_the_preview_sends_nothing_and_starts_nothing(
        self, qapp: QApplication, controller: MatchController, cloud: CloudRig
    ) -> None:
        previews: list[CloudPreview] = []
        panel = MatchPanel(controller, confirm_cloud=lambda p: previews.append(p) or False)
        recall(qapp, panel)
        panel.buttons["all"].click()
        panel.buttons["checklist"].click()
        wait_for(qapp, lambda: "cancelled" in panel.footer.text())
        assert len(previews) == 1
        assert "OpenRouter" in previews[0].destination
        assert cloud.inner.sent == []
        assert not cloud.consent.granted
        assert panel.pages.currentIndex() == PAGE_CANDIDATES

    def test_approving_grants_consent_and_runs_the_step(
        self, qapp: QApplication, controller: MatchController, cloud: CloudRig
    ) -> None:
        cloud.inner.json_fn = lambda _m: {
            "requirements": [{"text": "Python", "kind": "must", "weight": 1}]
        }
        panel = MatchPanel(controller, confirm_cloud=lambda _p: True)
        recall(qapp, panel)
        panel.buttons["checklist"].click()
        wait_for(qapp, lambda: panel.pages.currentIndex() == PAGE_CHECKLIST)
        assert cloud.consent.granted
        assert cloud.inner.sent

    def test_local_steps_never_show_the_dialog(
        self, qapp: QApplication, controller: MatchController, cloud: CloudRig
    ) -> None:
        cloud.router._settings = cloud.router._settings.model_copy(update={"routing": {}})
        shown: list[CloudPreview] = []
        panel = MatchPanel(controller, confirm_cloud=lambda p: shown.append(p) or True)
        recall(qapp, panel)
        panel.buttons["checklist"].click()
        wait_for(qapp, lambda: panel.pages.currentIndex() == PAGE_CHECKLIST)
        assert shown == []
