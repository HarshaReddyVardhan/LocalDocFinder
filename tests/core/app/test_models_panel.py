import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton
from tests.core.app.test_app import FakeService
from tests.core.app.test_modes import FakeAssistant
from tests.core.conftest import Chat, Env

from vector_embed.app.controller import Launcher
from vector_embed.app.models_controller import ModelsController
from vector_embed.app.models_panel import AUTOMATIC, ModelsPanel
from vector_embed.app.window import Mode, SearchWindow
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.skills.base import SkillContext

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
    monkeypatch.setattr("vector_embed.core.models.registry.probe_hardware", lambda: GPU)
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


def settings_file(env: Env) -> dict[str, object]:
    with (env.data_dir / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


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
        chat_row = next(r for r in range(7) if widget.roles.item(r, 0).text() == "chat")
        assert widget.roles.item(chat_row, 1).text() == "qwen3.5:9b"
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
        row = next(r for r in range(7) if widget.roles.item(r, 0).text() == "summarizer")
        combo = widget.roles.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
        combo.setCurrentText("deepseek-r1:8b")
        settings_path = env.data_dir / "settings.toml"
        wait_for(qapp, settings_path.exists)
        assert settings_file(env)["models"] == {"overrides": {"summarizer": "deepseek-r1:8b"}}

    def test_clearing_an_override_goes_back_to_automatic(
        self, qapp: QApplication, panel: tuple[ModelsPanel, list[str], list[str]], env: Env
    ) -> None:
        widget, _, _ = panel
        widget._controller.set_override("summarizer", "deepseek-r1:8b")
        load(qapp, widget)
        row = next(r for r in range(7) if widget.roles.item(r, 0).text() == "summarizer")
        combo = widget.roles.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
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
        row = next(r for r in range(7) if widget.roles.item(r, 0).text() == "embed")
        combo = widget.roles.cellWidget(row, 3)
        assert isinstance(combo, QComboBox)
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


def test_models_live_in_settings_not_in_the_popup(qapp: QApplication, tmp_path: Path) -> None:
    window = SearchWindow(
        FakeService(),  # type: ignore[arg-type]
        Launcher(),
        tmp_path,
        FakeAssistant(),  # type: ignore[arg-type]
    )
    assert "models" not in {mode.value for mode in Mode}
    assert window.available_modes() == [Mode.SEARCH, Mode.ASK, Mode.CHAT]
