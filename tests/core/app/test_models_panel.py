import time
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant
from tests.core.conftest import Chat, Env

from localdoc_finder.app.controller import Launcher
from localdoc_finder.app.models_controller import ModelsController
from localdoc_finder.app.models_panel import (
    AUTOMATIC,
    CHOICE_COLUMN,
    MODEL_COLUMN,
    ROLE_COLUMN,
    ModelsPanel,
    reason_text,
    recommendation_text,
)
from localdoc_finder.app.window import Mode, SearchWindow
from localdoc_finder.core.models.hardware import Hardware
from localdoc_finder.core.models.registry import BETTER_OPTION, ModelRegistry
from localdoc_finder.core.skills.base import SkillContext

GPU = Hardware("RTX 2070", 8192, 7000, 32000, 16000, 8, True)


def wait_for(qapp: QApplication, condition: Callable[[], bool], timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


@pytest.fixture
def controller(
    chat: Chat, skill_ctx: SkillContext, env: Env, monkeypatch: pytest.MonkeyPatch
) -> ModelsController:
    chat.client.models["deepseek-r1:8b"] = {"caps": ["completion"], "size": 5 * 1024**3}
    chat.client.models["mxbai-embed-large"] = {"caps": ["embedding"], "size": 700 * 1024**2}
    monkeypatch.setattr("localdoc_finder.core.models.registry.probe_hardware", lambda: GPU)
    chat.gateway._registry._probe = lambda: GPU
    return ModelsController(lambda: skill_ctx, env.data_dir / "settings.toml")


@pytest.fixture
def panel(
    qapp: QApplication, controller: ModelsController
) -> tuple[ModelsPanel, list[str], list[str]]:
    messages: list[str] = []
    asked: list[str] = []

    def confirm(message: str) -> bool:
        asked.append(message)
        return True

    widget = ModelsPanel(controller, confirm=confirm)
    widget.status_changed.connect(messages.append)
    return widget, messages, asked


def load(qapp: QApplication, widget: ModelsPanel) -> None:
    widget.refresh()
    wait_for(qapp, lambda: widget.models.rowCount() > 0 and widget.health_view.toPlainText() != "")


def role_row(widget: ModelsPanel, role: str) -> int:
    rows = range(widget.roles.rowCount())
    return next(
        r for r in rows if widget.roles.item(r, ROLE_COLUMN).data(Qt.ItemDataRole.UserRole) == role
    )


def choice(widget: ModelsPanel, role: str) -> QComboBox:
    combo = widget.roles.cellWidget(role_row(widget, role), CHOICE_COLUMN)
    assert isinstance(combo, QComboBox)
    return combo


def settings_file(env: Env) -> dict[str, object]:
    with (env.data_dir / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_the_controller_follows_a_rebuilt_context(
    chat: Chat, skill_ctx: SkillContext, env: Env
) -> None:
    # Regression: after a settings change the tab kept the first registry, so it showed (and
    # changed) models the popup no longer used.
    contexts = [skill_ctx]
    controller = ModelsController(lambda: contexts[-1], env.data_dir / "settings.toml")
    first = controller.registry
    assert controller.registry is first  # the same context keeps the same manager
    fresh = ModelRegistry(first.catalog, [])
    contexts.append(replace(skill_ctx, extras={**skill_ctx.extras, "models": fresh}))
    assert controller.registry is fresh


class TestRendering:
    def test_models_roles_and_flags_are_shown(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
    ) -> None:
        widget, messages, _ = panel
        load(qapp, widget)
        names = {widget.models.item(r, 0).text(): r for r in range(widget.models.rowCount())}
        assert {"qwen3.5:9b", "mxbai-embed-large", "deepseek-r1:8b"} <= set(names)
        assert "512-token limit" in widget.models.item(names["mxbai-embed-large"], 3).text()
        assert "unused" in widget.models.item(names["deepseek-r1:8b"], 3).text()
        assert widget.models.item(names["qwen3.5:9b"], 3).text() == "ok"
        assert "RTX 2070" in widget.hardware.text()
        assert widget.roles.rowCount() == 7
        chat_row = role_row(widget, "chat")
        assert widget.roles.item(chat_row, ROLE_COLUMN).text() == "Chat"
        assert widget.roles.item(chat_row, MODEL_COLUMN).text() == "qwen3.5:9b"
        assert "Ask" in widget.roles.item(chat_row, 1).text()  # says which features use it
        assert widget.roles.item(chat_row, 3).text() == "automatic"
        assert any("models installed" in m for m in messages)

    def test_health_tab_shows_the_dashboard_text(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
    ) -> None:
        widget, _, _ = panel
        load(qapp, widget)
        text = widget.health_view.toPlainText()
        assert "queue           :" in text
        assert "cloud spend" in text

    def test_recommendations_offer_a_pull_button(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], chat: Chat
    ) -> None:
        widget, _, _ = panel
        del chat.client.models["qwen3.5:9b"]
        load(qapp, widget)
        buttons = [b for b in widget.findChildren(QPushButton) if b.text().startswith("Pull ")]
        assert any("qwen3.5:9b" in b.text() for b in buttons)

    def test_nothing_to_suggest_message(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], chat: Chat
    ) -> None:
        widget, _, _ = panel
        for name in ("qwen2.5vl:3b", "qwen2.5:7b", "dengcao/Qwen3-Reranker-0.6B:Q8_0"):
            chat.client.models[name] = {"caps": ["completion"]}
        load(qapp, widget)
        # With everything installed the layout holds either the buttons or the all-clear note.
        assert widget.recommendations.count() >= 1


class TestActions:
    def test_pull_shows_progress_then_refreshes(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], chat: Chat
    ) -> None:
        widget, messages, _ = panel
        load(qapp, widget)
        chat.client.models["qwen3:8b"] = {"caps": ["completion"]}
        widget.pull("qwen3:8b")
        widget.pull("qwen3:8b")  # a second click while pulling is ignored
        wait_for(qapp, lambda: "pulled qwen3:8b" in messages)
        assert "pulled qwen3:8b" in messages
        assert any(m.startswith("pulling") for m in messages)
        assert not widget.progress.isVisible()
        wait_for(
            qapp,
            lambda: any(
                widget.models.item(r, 0).text() == "qwen3:8b"
                for r in range(widget.models.rowCount())
            ),
        )
        assert [c[0] for c in chat.client.calls].count("pull") == 1

    def test_role_override_is_saved(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], env: Env
    ) -> None:
        widget, _, _ = panel
        load(qapp, widget)
        combo = choice(widget, "summarizer")
        combo.setCurrentText("deepseek-r1:8b")
        settings_path = env.data_dir / "settings.toml"
        wait_for(qapp, settings_path.exists)
        assert settings_file(env)["models"] == {"overrides": {"summarizer": "deepseek-r1:8b"}}

    def test_role_choices_offer_only_models_that_can_do_the_job(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
    ) -> None:
        widget, _, _ = panel
        load(qapp, widget)
        options = {}
        for role in ("chat", "embed"):
            combo = choice(widget, role)
            options[role] = {combo.itemText(i) for i in range(combo.count())}
        assert "deepseek-r1:8b" in options["chat"]
        assert not {"qwen3-embedding:0.6b", "mxbai-embed-large"} & options["chat"]
        assert "mxbai-embed-large" in options["embed"]
        assert "deepseek-r1:8b" not in options["embed"]

    def test_clearing_an_override_goes_back_to_automatic(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], env: Env
    ) -> None:
        widget, _, _ = panel
        widget._controller.set_override("summarizer", "deepseek-r1:8b")
        load(qapp, widget)
        combo = choice(widget, "summarizer")
        assert combo.currentText() == "deepseek-r1:8b"
        combo.setCurrentText(AUTOMATIC)
        wait_for(qapp, lambda: settings_file(env)["models"] == {"overrides": {}})
        assert settings_file(env)["models"] == {"overrides": {}}

    def test_switching_the_embedder_asks_first_and_schedules_a_rescan(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], env: Env
    ) -> None:
        widget, messages, asked = panel
        env.state.manifest_set("a", 1, 1, "h")
        load(qapp, widget)
        combo = choice(widget, "embed")
        combo.setCurrentText("mxbai-embed-large")
        assert asked and "re-indexing" in asked[0]
        assert settings_file(env)["embedding"] == {"model": "mxbai-embed-large"}
        assert env.state.get_meta("last_reconcile") == "0"
        assert any("rebuilt" in m for m in messages)

    def test_declining_the_embedder_switch_changes_nothing(
        self, qapp: QApplication, controller: ModelsController, env: Env
    ) -> None:
        widget = ModelsPanel(controller, confirm=lambda _m: False)
        load(qapp, widget)
        widget._on_role_choice("embed", "mxbai-embed-large")
        assert not (env.data_dir / "settings.toml").exists()

    def test_choosing_the_current_embedder_is_ignored(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], env: Env
    ) -> None:
        widget, _, asked = panel
        load(qapp, widget)
        widget._on_role_choice("embed", "qwen3-embedding:0.6b")
        assert asked == []

    def test_failures_are_reported_and_clear_the_progress_bar(
        self,
        qapp: QApplication,
        panel: tuple[ModelsPanel, list[str], list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        widget, messages, _ = panel

        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr(widget._controller, "pull", boom)
        widget.pull("x:1")
        wait_for(qapp, lambda: "disk full" in messages)
        assert "disk full" in messages
        widget.pull("x:1")  # pulling again is allowed after a failure
        wait_for(qapp, lambda: messages.count("disk full") == 2)

    def test_activate_and_deactivate_drive_the_refresh_timer(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
    ) -> None:
        widget, _, _ = panel
        widget.activate()
        assert widget._timer.isActive()
        widget.deactivate()
        assert not widget._timer.isActive()
        wait_for(qapp, lambda: widget.models.rowCount() > 0)


class TestHealthRefresh:
    def test_refreshes_do_not_stack_up_behind_a_slow_server(
        self,
        qapp: QApplication,
        panel: tuple[ModelsPanel, list[str], list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import threading

        widget, _, _ = panel
        release = threading.Event()
        calls: list[int] = []

        def slow() -> str:
            calls.append(1)
            release.wait(5)
            return "health text"

        monkeypatch.setattr(widget._controller, "health_text", slow)
        for _ in range(5):  # the timer fires while the first call is still waiting
            widget.refresh_health()
        wait_for(qapp, lambda: calls)
        release.set()
        wait_for(qapp, lambda: widget.health_view.toPlainText() == "health text")
        assert calls == [1]
        widget.refresh_health()  # and a fresh one is allowed once it finished
        wait_for(qapp, lambda: len(calls) == 2)

    def test_a_health_failure_is_shown_there_and_does_not_cancel_a_pull(
        self,
        qapp: QApplication,
        panel: tuple[ModelsPanel, list[str], list[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        widget, _, _ = panel
        widget._pulling.add("big-model:70b")  # a download is in progress
        widget.progress.setVisible(True)

        def boom() -> str:
            raise RuntimeError("ollama is not answering")

        monkeypatch.setattr(widget._controller, "health_text", boom)
        widget.refresh_health()
        wait_for(qapp, lambda: "not answering" in widget.health_view.toPlainText())
        assert "not answering" in widget.health_view.toPlainText()
        assert widget._pulling == {"big-model:70b"}  # the download keeps its place
        assert not widget.progress.isHidden()  # still showing the download
        widget.refresh_health()  # a failure also frees the slot for the next refresh
        wait_for(qapp, lambda: not widget._health_running)

    def test_the_timer_follows_visibility(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
    ) -> None:
        widget, _, _ = panel
        assert not widget._timer.isActive()
        widget.show()
        assert widget._timer.isActive()
        widget.hide()
        assert not widget._timer.isActive()

    def test_the_interval_is_a_glance_not_a_monitor(self) -> None:
        from localdoc_finder.app import models_panel

        assert models_panel.HEALTH_REFRESH_MS >= 10_000


def test_models_live_in_settings_not_in_the_popup(qapp: QApplication, tmp_path: Path) -> None:
    window = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        Launcher(),
        tmp_path,
        FakeAssistant(),  # type: ignore[arg-type]
    )
    assert "models" not in {mode.value for mode in Mode}
    assert window.available_modes() == [Mode.SEARCH, Mode.ASK, Mode.CHAT]


def test_the_mouse_wheel_never_changes_a_role_choice(
    qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]]
) -> None:
    widget, _, _ = panel
    load(qapp, widget)
    combo = choice(widget, "chat")
    before = combo.currentText()
    wheel = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(combo, wheel)
    assert combo.currentText() == before
    assert not wheel.isAccepted()  # handed on, so the table scrolls instead


def test_reasons_are_said_in_words() -> None:
    assert reason_text("override") == "you chose it"
    assert reason_text("qwen3-embedding:0.6b cannot serve chat; preferred") == (
        "qwen3-embedding:0.6b cannot serve chat; automatic"
    )
    assert reason_text("no installed model fits") == "no installed model fits"


def test_recommendations_name_the_role_in_words() -> None:
    assert recommendation_text("chat", "llama3.2", BETTER_OPTION) == (
        "Chat: llama3.2 would be a better fit"
    )
    assert recommendation_text("embed", "bge-m3", "pull it; re-index required").endswith(
        "(pull it; re-index required)"
    )
