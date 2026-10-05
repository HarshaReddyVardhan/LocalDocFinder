import os
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit, QMenu
from tests.core.app.test_models_panel import GPU, wait_for
from tests.core.conftest import Chat, Env

from localdoc_finder.app import main as app_main
from localdoc_finder.app.cloud_tab import CloudTab
from localdoc_finder.app.models_controller import ModelsController
from localdoc_finder.app.settings_controller import NO_UPDATES, SettingsController, app_version
from localdoc_finder.app.settings_tabs import AboutTab, GeneralTab, IndexingTab
from localdoc_finder.app.settings_window import MIN_WINDOW_SIZE, SettingsWindow
from localdoc_finder.app.theme import (
    Scheme,
    apply_theme,
    palette_for,
    scheme_in_use,
    secondary_text,
)
from localdoc_finder.core.indexing_control import IndexingStatus, StartResult
from localdoc_finder.core.models.benchmark import BenchKind, BenchResult, record_result
from localdoc_finder.core.models.hardware import Load
from localdoc_finder.core.settings import (
    CloudProviderSettings,
    Settings,
    SettingsError,
    load_settings,
)
from localdoc_finder.core.settings_io import set_setting
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.updates import UpdateKind, Updater

SETTINGS_MIN_WIDTH, SETTINGS_MIN_HEIGHT = MIN_WINDOW_SIZE


class FakeKeys:
    def __init__(self) -> None:
        self.keys: dict[str, str] = {}

    def get(self, provider: str) -> str | None:
        return self.keys.get(provider)

    def set(self, provider: str, key: str) -> None:
        self.keys[provider] = key

    def delete(self, provider: str) -> None:
        self.keys.pop(provider, None)


@pytest.fixture
def keys() -> FakeKeys:
    return FakeKeys()


@pytest.fixture
def autostart() -> list[bool]:
    return []


@pytest.fixture
def controller(env: Env, keys: FakeKeys, autostart: list[bool]) -> SettingsController:
    return SettingsController(
        env.data_dir / "settings.toml", env.state, keys, apply_autostart=autostart.append
    )


def saved(env: Env) -> dict[str, object]:
    with (env.data_dir / "settings.toml").open("rb") as handle:
        return tomllib.load(handle)


# ------------------------------------------------------------------ controller
def test_hotkey_is_validated_and_normalised(env: Env, controller: SettingsController) -> None:
    assert controller.set_hotkey(" Ctrl + Alt + F9 ") == "ctrl+alt+f9"
    assert saved(env)["search"] == {"hotkey": "ctrl+alt+f9"}
    with pytest.raises(SettingsError, match="invalid hotkey"):
        controller.set_hotkey("ctrl+banana")
    assert controller.settings().search.hotkey == "ctrl+alt+f9"


def test_a_chosen_scope_needs_a_folder_and_dedupes(
    env: Env, controller: SettingsController
) -> None:
    controller.set_scope("chosen", [r"D:\a", r"D:\a", r"D:\b"], "documents")
    scope = controller.settings().scope
    assert (scope.coverage, scope.roots) == ("chosen", (r"D:\a", r"D:\b"))
    with pytest.raises(SettingsError, match="at least one"):
        controller.set_scope("chosen", [], "documents")


def test_system_folders_cannot_be_chosen(env: Env, controller: SettingsController) -> None:
    system = os.environ.get("SYSTEMROOT", r"C:\Windows")
    with pytest.raises(SettingsError, match="never indexed"):
        controller.set_scope("chosen", [system], "documents")


def test_custom_kinds_are_saved_and_cleared(controller: SettingsController) -> None:
    controller.set_scope("entire_pc", [], "custom", custom_kinds=["images", "code"])
    scope = controller.settings().scope
    assert (scope.file_types, scope.custom_kinds) == ("custom", {"images", "code"})
    controller.set_scope("entire_pc", [], "everything", custom_kinds=["images"])
    assert controller.settings().scope.file_types == "everything"


@pytest.mark.parametrize(("kinds", "message"), [([], "at least one"), (["videos"], "unknown")])
def test_bad_custom_kinds_are_refused(
    controller: SettingsController, kinds: list[str], message: str
) -> None:
    with pytest.raises(SettingsError, match=message):
        controller.set_scope("entire_pc", [], "custom", custom_kinds=kinds)


def test_extra_folders_are_kept_on_top_of_the_whole_pc(
    env: Env, controller: SettingsController
) -> None:
    controller.set_scope("chosen", [r"D:\a"], "everything")
    controller.set_scope("entire_pc", [r"E:\External"], "documents")
    scope = controller.settings().scope
    assert (scope.coverage, scope.roots, scope.file_types) == (
        "entire_pc",
        (r"E:\External",),
        "documents",
    )
    controller.set_scope("entire_pc", [], "documents")
    assert controller.settings().scope.roots == ()


def test_autostart_is_saved_and_applied(
    controller: SettingsController, autostart: list[bool]
) -> None:
    controller.set_start_with_windows(False)
    assert autostart == [False]
    assert not controller.settings().app.start_with_windows


def test_privacy_budget_and_updates_settings(env: Env, controller: SettingsController) -> None:
    controller.set_redact_personal(True)
    controller.set_mask_ids_locally(True)
    controller.set_monthly_budget(12.5)
    controller.set_auto_check(False)
    settings = controller.settings()
    assert settings.privacy.redact_personal
    assert settings.privacy.mask_ids_locally
    assert settings.cloud.monthly_budget_usd == 12.5
    assert not settings.updates.auto_check
    controller.set_monthly_budget(None)
    assert controller.settings().cloud.monthly_budget_usd is None
    with pytest.raises(SettingsError):
        controller.set_monthly_budget(0)


def test_key_statuses_never_expose_the_key(
    env: Env, controller: SettingsController, keys: FakeKeys
) -> None:
    provider = CloudProviderSettings(base_url="https://x.test/v1", label="X Cloud")
    set_setting(
        env.data_dir / "settings.toml",
        ["cloud", "providers", "x"],
        provider.model_dump(mode="json"),
    )
    assert [(s.label, s.has_key) for s in controller.key_statuses()] == [("X Cloud", False)]
    controller.set_key("x", "sk-secret")
    assert [s.has_key for s in controller.key_statuses()] == [True]
    assert "sk-secret" not in repr(controller.key_statuses())
    controller.delete_key("x")
    assert [s.has_key for s in controller.key_statuses()] == [False]


def test_speed_tests_and_update_check(env: Env, controller: SettingsController) -> None:
    assert controller.speed_tests() == []
    result = BenchResult("m", BenchKind.CHAT, 20.0, 1.0)
    record_result(env.state, result)
    assert controller.speed_tests() == [result]
    outcome = controller.check_now()
    assert (outcome.kind, outcome.message) == (UpdateKind.NOT_CONFIGURED, NO_UPDATES)
    assert app_version()


# ------------------------------------------------------------------ window
@pytest.fixture
def models(
    chat: Chat, skill_ctx: SkillContext, env: Env, monkeypatch: pytest.MonkeyPatch
) -> ModelsController:
    monkeypatch.setattr("localdoc_finder.core.models.registry.probe_hardware", lambda: GPU)
    monkeypatch.setattr("localdoc_finder.core.health.probe_hardware", lambda: GPU)
    monkeypatch.setattr("localdoc_finder.core.health.probe_load", lambda: Load(10.0, 20))
    chat.gateway._registry._probe = lambda: GPU
    return ModelsController(lambda: skill_ctx, env.data_dir / "settings.toml")


@pytest.fixture
def window(
    qapp: QApplication, controller: SettingsController, models: ModelsController
) -> SettingsWindow:
    folders: list[str] = []
    win = SettingsWindow(
        controller,
        models,
        general=GeneralTab(controller, choose_folder=lambda: folders.pop() if folders else None),
        cloud=CloudTab(controller, ask_key=lambda provider: "sk-typed"),
    )
    win._folders = folders  # type: ignore[attr-defined]  # test hook for the folder chooser
    return win


def test_window_has_the_expected_tabs(window: SettingsWindow) -> None:
    # Regression: a bare & was read as a shortcut marker and shown as "Models _Health".
    titles = [window.tabs.tabText(i).replace("&&", "&") for i in range(window.tabs.count())]
    assert titles == [
        "General",
        "Features",
        "Models & Health",
        "Cloud & Privacy",
        "Updates",
        "Advanced",
        "About",
    ]


def test_advanced_tab_saves_each_kind_of_option(
    window: SettingsWindow, controller: SettingsController
) -> None:
    tab = window.advanced
    results = tab.editors[("search", "results")]
    assert isinstance(results, QLineEdit)
    assert results.text() == str(controller.settings().search.results)
    results.setText("12")
    results.editingFinished.emit()
    assert controller.settings().search.results == 12
    assert "Results saved" in window.status.text()

    captions = tab.editors[("images", "enable_captions")]
    assert isinstance(captions, QCheckBox)
    captions.click()
    assert controller.settings().images.enable_captions is captions.isChecked()

    level = tab.editors[("log_level",)]
    assert isinstance(level, QComboBox)
    level.setCurrentText("DEBUG")
    level.activated.emit(level.currentIndex())
    assert controller.settings().log_level == "DEBUG"


def test_advanced_tab_rejects_an_invalid_value_and_shows_the_real_one(
    window: SettingsWindow, controller: SettingsController
) -> None:
    results = window.advanced.editors[("search", "results")]
    assert isinstance(results, QLineEdit)
    before = controller.settings().search.results
    for bad in ("many", "0"):  # not a number; a number the schema forbids (must be > 0)
        results.setText(bad)
        results.editingFinished.emit()
        assert controller.settings().search.results == before
        assert results.text() == str(before)
    assert window.status.text()


def test_general_tab_applies_a_hotkey(
    window: SettingsWindow, env: Env, controller: SettingsController
) -> None:
    seen: list[str] = []
    window.hotkey_changed.connect(seen.append)
    window.general.hotkey.setText("ctrl+alt+f9")
    window.general.apply_hotkey.click()
    assert seen == ["ctrl+alt+f9"]
    assert controller.settings().search.hotkey == "ctrl+alt+f9"
    assert "hotkey set to ctrl+alt+f9" in window.status.text()


def test_general_tab_reports_a_bad_hotkey_without_saving(
    window: SettingsWindow, controller: SettingsController
) -> None:
    seen: list[str] = []
    window.hotkey_changed.connect(seen.append)
    window.general.hotkey.setText("ctrl+banana")
    window.general.apply_hotkey.click()
    assert seen == []
    assert "invalid hotkey" in window.status.text()
    assert controller.settings().search.hotkey == "ctrl+alt+space"


def test_general_tab_saves_a_chosen_scope(
    window: SettingsWindow, controller: SettingsController
) -> None:
    editor = window.general.scope
    window._folders.append(r"D:\Work")  # type: ignore[attr-defined]
    editor.add_folder.click()
    editor.chosen.setChecked(True)
    window.general.save_scope.click()
    scope = controller.settings().scope
    assert (scope.coverage, scope.roots) == ("chosen", (r"D:\Work",))
    editor.folders.setCurrentRow(0)
    editor.remove_folder.click()
    assert not editor.is_valid()
    assert not window.general.save_scope.isEnabled()


def test_general_tab_shows_the_saved_scope(
    window: SettingsWindow, controller: SettingsController
) -> None:
    controller.set_scope("chosen", [r"D:\only"], "everything")
    window.general.refresh()
    editor = window.general.scope
    assert editor.chosen.isChecked()
    assert editor.roots() == [r"D:\only"]
    assert editor.choice().file_types == "everything"


def test_autostart_checkbox_saves_and_applies(
    window: SettingsWindow, controller: SettingsController, autostart: list[bool]
) -> None:
    window.general.start_with_windows.click()
    assert autostart == [False]
    assert not controller.settings().app.start_with_windows


def test_cloud_tab_toggles_and_budget(
    window: SettingsWindow, controller: SettingsController
) -> None:
    cloud = window.cloud
    cloud.redact.click()
    cloud.mask_local.click()
    cloud.limit.click()
    cloud.budget.setValue(25.0)
    cloud.budget.editingFinished.emit()
    settings = controller.settings()
    assert settings.privacy.redact_personal
    assert settings.privacy.mask_ids_locally
    assert settings.cloud.monthly_budget_usd == 25.0
    cloud.limit.click()
    assert controller.settings().cloud.monthly_budget_usd is None


def test_cloud_tab_stores_a_key_without_showing_it(
    env: Env, window: SettingsWindow, controller: SettingsController, keys: FakeKeys
) -> None:
    provider = CloudProviderSettings(base_url="https://x.test/v1", label="X Cloud")
    set_setting(
        env.data_dir / "settings.toml",
        ["cloud", "providers", "x"],
        provider.model_dump(mode="json"),
    )
    window.cloud.refresh()
    assert window.cloud.key_labels[0].text() == "X Cloud: no key"
    window.cloud.key_buttons[0].click()
    assert keys.keys == {"x": "sk-typed"}
    assert window.cloud.key_labels[0].text() == "X Cloud: key stored"
    assert "sk-typed" not in window.cloud.key_labels[0].text()


def test_updates_tab_without_an_update_source(
    qapp: QApplication, window: SettingsWindow, controller: SettingsController
) -> None:
    window.updates.check_now.click()
    assert not window.updates.check_now.isEnabled()  # busy while the check runs
    wait_for(qapp, window.updates.check_now.isEnabled)
    assert window.updates.result.text() == NO_UPDATES
    assert window.updates.restart.isHidden()
    window.updates.auto_check.click()
    assert not controller.settings().updates.auto_check


class FakeUpdateManager:
    def __init__(self) -> None:
        self.applied: list[object] = []

    def get_current_version(self) -> str:
        return "1.0.0"

    def check_for_updates(self) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(TargetFullRelease=SimpleNamespace(Version="1.1.0"))

    def download_updates(self, info: object, progress: object = None) -> None:
        return None

    def apply_updates_and_restart(self, update: object) -> None:
        self.applied.append(update)


def test_updates_tab_offers_a_restart_when_an_update_is_ready(
    qapp: QApplication, env: Env, keys: FakeKeys, models: ModelsController
) -> None:
    manager = FakeUpdateManager()
    updater = Updater("https://github.com/x/y", state=env.state, factory=lambda _url: manager)
    controller = SettingsController(
        env.data_dir / "settings.toml", env.state, keys, updater=updater
    )
    window = SettingsWindow(controller, models)
    window.updates.check_now.click()
    wait_for(qapp, lambda: not window.updates.restart.isHidden())
    assert window.updates.result.text() == "Version 1.1.0 is ready. Restart to update."
    window.updates.restart.click()
    assert len(manager.applied) == 1


def test_restart_without_an_updater_is_refused(
    controller: SettingsController, window: SettingsWindow
) -> None:
    with pytest.raises(RuntimeError, match="not available"):
        controller.restart_to_update()
    window.updates.restart.click()  # the tab reports it instead of raising
    assert "not available" in window.status.text()


def test_about_tab_shows_the_data_folder(window: SettingsWindow, env: Env) -> None:
    assert window.about.data_folder.text() == str(env.data_dir)
    assert window.about.version.text().startswith("LocalDoc Finder ")


def deleting_window(
    env: Env,
    keys: FakeKeys,
    models: ModelsController,
    autostart: list[bool],
    confirm: bool,
) -> tuple[SettingsWindow, list[str], list[Path], list[int]]:
    steps: list[str] = []
    deleted: list[Path] = []
    quits: list[int] = []
    controller = SettingsController(
        env.data_dir / "settings.toml",
        env.state,
        keys,
        apply_autostart=lambda enabled: steps.append(f"autostart {enabled}"),
        stop_others=lambda: steps.append("stop"),
        schedule_deletion=deleted.append,
    )
    window = SettingsWindow(
        controller, models, about=AboutTab(controller, lambda: confirm, lambda: quits.append(1))
    )
    return window, steps, deleted, quits


def test_delete_my_data_stops_everything_schedules_the_wipe_and_quits(
    env: Env, keys: FakeKeys, models: ModelsController, autostart: list[bool]
) -> None:
    window, steps, deleted, quits = deleting_window(env, keys, models, autostart, confirm=True)
    set_setting(
        env.data_dir / "settings.toml",
        ["cloud", "providers", "x"],
        CloudProviderSettings(base_url="https://x.test/v1").model_dump(mode="json"),
    )
    keys.set("x", "sk-secret")
    window.about.delete_data.click()
    assert keys.get("x") is None  # stored API keys go with the data
    assert steps == ["autostart False", "stop"]
    assert deleted == [env.data_dir]
    assert quits == [1]


def test_delete_my_data_refuses_a_foreign_folder_before_stopping_anything(
    env: Env, keys: FakeKeys, models: ModelsController, autostart: list[bool]
) -> None:
    window, steps, deleted, quits = deleting_window(env, keys, models, autostart, confirm=True)
    env.state.data_dir = env.data_dir / "elsewhere"  # simulate a state db pointing elsewhere
    window.about.delete_data.click()
    assert (steps, deleted, quits) == ([], [], [])


def test_delete_my_data_needs_confirmation(
    env: Env, keys: FakeKeys, models: ModelsController, autostart: list[bool]
) -> None:
    window, steps, deleted, quits = deleting_window(env, keys, models, autostart, confirm=False)
    window.about.delete_data.click()
    assert (steps, deleted, quits) == ([], [], [])


def test_models_tab_lists_speed_tests(qapp: QApplication, window: SettingsWindow, env: Env) -> None:
    window.models.refresh()
    assert "Not measured yet" in window.models.speed.text()
    record_result(env.state, BenchResult("qwen3.5:9b", BenchKind.CHAT, 3.0, 1.0))
    record_result(env.state, BenchResult("qwen3-embedding:0.6b", BenchKind.EMBED, 90.0, 0.5))
    window.models.refresh()
    text = window.models.speed.text()
    assert "qwen3.5:9b: 3.0 tok/s  (slow on this machine)" in text
    assert "qwen3-embedding:0.6b: 90.0 texts/s" in text
    wait_for(qapp, lambda: window.models.panel.models.rowCount() > 0)
    window.models.panel.deactivate()


def test_open_refreshes_and_shows(
    window: SettingsWindow, controller: SettingsController, qapp: QApplication
) -> None:
    controller.set_hotkey("ctrl+alt+f8")
    window.open()
    assert window.isVisible()
    assert window.general.hotkey.text() == "ctrl+alt+f8"
    window.close()


# ------------------------------------------------------------------ app wiring
def test_build_settings_window_forwards_hotkey_changes(
    qapp: QApplication,
    env: Env,
    skill_ctx: SkillContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_main, "KeyringStore", FakeKeys)
    seen: list[str] = []
    window = app_main.build_settings_window(env.settings, env.state, lambda: skill_ctx, seen.append)
    window.hotkey_changed.emit("ctrl+alt+f7")
    assert seen == ["ctrl+alt+f7"]


@pytest.mark.parametrize("registers", [True, False])
def test_hotkey_applier_reregisters_and_updates_the_tooltip(registers: bool) -> None:
    calls: list[str] = []

    class FakeHotkey:
        spec: str | None = "ctrl+alt+space"

        def register(self, spec: str) -> bool:
            calls.append(spec)
            if spec == "bad":
                raise ValueError(spec)
            if registers:
                self.spec = spec
            return registers

    class FakeTray:
        tip = ""

        def setToolTip(self, text: str) -> None:  # noqa: N802
            self.tip = text

    tray = FakeTray()
    apply: Callable[[str], None] = app_main.hotkey_applier(FakeHotkey(), tray)  # type: ignore[arg-type]
    apply("ctrl+alt+f6")
    assert calls == ["ctrl+alt+f6"]  # no separate unregister: register() swaps safely
    expected = (
        "LocalDoc Finder (ctrl+alt+f6)"
        if registers
        else "LocalDoc Finder (ctrl+alt+space) - ctrl+alt+f6 is unavailable"
    )
    assert tray.tip == expected
    apply("bad")
    assert tray.tip.endswith("bad is unavailable")


def test_settings_defaults_include_the_new_sections() -> None:
    settings = Settings()
    assert settings.app.start_with_windows
    assert settings.updates.auto_check


# ------------------------------------------------------------------ changes apply without a restart
def test_every_saved_setting_notifies_the_app(env: Env, keys: FakeKeys) -> None:
    changes: list[int] = []
    controller = SettingsController(
        env.data_dir / "settings.toml", env.state, keys, on_changed=lambda: changes.append(1)
    )
    controller.set_redact_personal(True)
    controller.set_mask_ids_locally(True)
    controller.set_monthly_budget(5.0)
    controller.set_start_with_windows(False)
    controller.set_hotkey("ctrl+alt+f6")
    controller.set_scope("chosen", [r"D:\a"], "documents")
    controller.set_key("x", "sk-secret")
    controller.delete_key("x")
    assert len(changes) == 8


def test_a_rejected_change_does_not_notify(env: Env, keys: FakeKeys) -> None:
    changes: list[int] = []
    controller = SettingsController(
        env.data_dir / "settings.toml", env.state, keys, on_changed=lambda: changes.append(1)
    )
    with pytest.raises(SettingsError):
        controller.set_monthly_budget(-1.0)
    with pytest.raises(SettingsError):
        controller.set_scope("chosen", [], "documents")
    assert changes == []


def test_the_context_is_rebuilt_from_the_saved_settings_after_a_change(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[bool] = []

    def fake_build(settings: object, state: object, **_kw: object) -> object:
        built.append(settings.privacy.redact_personal)  # type: ignore[attr-defined]
        return object()

    monkeypatch.setattr(app_main.runtime, "build_skill_context", fake_build)
    path = env.data_dir / "settings.toml"
    factory = app_main.make_context_factory(env.settings, env.state, lambda: load_settings(path))
    first = factory()
    assert factory() is first  # cached until something changes
    set_setting(path, ["privacy", "redact_personal"], True)
    factory.invalidate()
    assert factory() is not first
    assert built == [False, True]  # the rebuilt context saw the new privacy setting


def test_one_consent_object_outlives_every_context_rebuild(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def fake_build(settings: object, state: object, **kw: object) -> object:
        seen.append(kw["consent"])
        return object()

    monkeypatch.setattr(app_main.runtime, "build_skill_context", fake_build)
    factory = app_main.make_context_factory(env.settings, env.state, None)
    factory()
    factory.consent.grant_session()  # the user ticked "don't ask again"
    factory.invalidate()  # a setting changed, so the context is rebuilt
    factory()
    assert seen == [factory.consent, factory.consent]
    assert factory.consent.session_granted  # still agreed: only a restart (or Forget) ends it


def test_an_invalid_settings_file_keeps_the_working_context_settings(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_main.runtime, "build_skill_context", lambda *_a, **_k: object())

    def broken() -> object:
        raise SettingsError("bad file")

    factory = app_main.make_context_factory(env.settings, env.state, broken)  # type: ignore[arg-type]
    factory.invalidate()  # must not raise
    assert factory() is not None


# ------------------------------------------------------------------ indexing tab and tray entry
class FakeIndexing:
    def __init__(self, waiting: int = 5) -> None:
        self.running = False
        self.paused = False
        self.waiting = waiting
        self.log: list[str] = []

    def status(self) -> IndexingStatus:
        return IndexingStatus(10, self.waiting, self.running, self.paused)

    def start(self) -> StartResult:
        self.log.append("start")
        self.running, self.paused = True, False
        return StartResult(True, "started")

    def pause(self) -> str:
        self.log.append("pause")
        self.paused = True
        return "pausing"


def test_indexing_tab_starts_and_pauses_and_shows_progress(
    qapp: QApplication, controller: SettingsController
) -> None:
    fake = FakeIndexing()
    tab = IndexingTab(controller, fake)  # type: ignore[arg-type]
    messages: list[str] = []
    tab.message.connect(messages.append)
    tab.refresh()
    assert tab.start.isEnabled()
    assert not tab.pause.isEnabled()
    assert tab.bar.value() == 66
    tab.start.click()
    assert tab.pause.isEnabled()
    assert not tab.start.isEnabled()
    tab.pause.click()
    assert fake.log == ["start", "pause"]
    assert messages == ["started", "pausing"]
    assert "paused" in tab.status.text()


def test_indexing_tab_start_button_is_disabled_once_nothing_is_waiting(
    qapp: QApplication, controller: SettingsController
) -> None:
    fake = FakeIndexing(waiting=0)
    tab = IndexingTab(controller, fake)  # type: ignore[arg-type]
    tab.refresh()
    assert not tab.start.isEnabled()


def test_settings_window_shows_the_indexing_tab_only_with_a_control(
    qapp: QApplication, controller: SettingsController, models: ModelsController
) -> None:
    without = SettingsWindow(controller, models)
    assert without.indexing is None
    with_it = SettingsWindow(controller, models, indexing=FakeIndexing())  # type: ignore[arg-type]
    assert with_it.indexing is not None


def test_tray_menu_toggles_start_and_pause(qapp: QApplication) -> None:
    fake = FakeIndexing()
    shown: list[str] = []
    menu = QMenu()
    app_main.add_indexing_actions(menu, lambda: fake, shown.append)  # type: ignore[arg-type,return-value]
    status_line, toggle = menu.actions()
    menu.aboutToShow.emit()
    assert toggle.text() == "Start indexing"
    assert "10 files" in status_line.text()
    toggle.trigger()
    menu.aboutToShow.emit()
    assert toggle.text() == "Pause indexing"
    toggle.trigger()
    assert fake.log == ["start", "pause"]
    assert shown == ["started", "pausing"]


# ------------------------------------------------------------------ resizing and theme
def test_window_can_shrink_below_its_content(window: SettingsWindow) -> None:
    """A tab's content scrolls; it must never set a minimum size bigger than the default."""
    assert window.minimumSizeHint().width() <= SETTINGS_MIN_WIDTH
    assert window.minimumSizeHint().height() <= SETTINGS_MIN_HEIGHT
    window.resize(SETTINGS_MIN_WIDTH, SETTINGS_MIN_HEIGHT)
    assert window.width() == SETTINGS_MIN_WIDTH


def test_theme_is_saved_and_validated(env: Env, controller: SettingsController) -> None:
    controller.set_theme("dark")
    assert saved(env)["app"] == {"theme": "dark"}
    with pytest.raises(SettingsError, match="unknown theme"):
        controller.set_theme("purple")


def test_theme_combo_saves_and_shows_the_choice(
    env: Env, window: SettingsWindow, controller: SettingsController
) -> None:
    assert window.general.theme.currentData() == "light"  # the default
    window.general.theme.setCurrentIndex(window.general.theme.findData("dark"))
    window.general.theme.activated.emit(window.general.theme.currentIndex())
    assert saved(env)["app"]["theme"] == "dark"  # type: ignore[index]  # TOML table
    window.general.refresh()
    assert window.general.theme.currentData() == "dark"


def test_apply_theme_pins_a_light_or_dark_palette(qapp: QApplication) -> None:
    original = qapp.palette()
    try:
        assert apply_theme(qapp, "dark") is Scheme.DARK
        assert scheme_in_use() is Scheme.DARK
        assert apply_theme(qapp, "light") is Scheme.LIGHT
        assert scheme_in_use() is Scheme.LIGHT
    finally:
        qapp.setPalette(original)


def test_secondary_text_fades_toward_the_background(qapp: QApplication) -> None:
    palette = palette_for(Scheme.LIGHT)
    assert secondary_text(palette, 1).name() == "#1b1b1f"
    assert secondary_text(palette, 0).name() == "#ffffff"


# ------------------------------------------------------------------ opening on the Cloud tab
def test_opening_on_a_given_tab_shows_it(window: SettingsWindow) -> None:
    window.open(window.cloud)
    assert window.tabs.currentIndex() == window._pages.index(window.cloud)
    assert window.tabs.tabText(window.tabs.currentIndex()).replace("&&", "&") == "Cloud & Privacy"
    window.open()  # without a page it stays where it was
    assert window.tabs.currentIndex() == window._pages.index(window.cloud)


def test_the_settings_opener_builds_the_window_once_and_can_open_the_cloud_tab(
    window: SettingsWindow,
) -> None:
    built: list[int] = []

    def build() -> SettingsWindow:
        built.append(1)
        return window

    opener = app_main.SettingsOpener(build)
    assert built == []  # nothing is built until it is needed
    opener.open(True)  # a menu action passes ``checked``
    opener.open_cloud()
    assert built == [1]
    assert window.tabs.currentIndex() == window._pages.index(window.cloud)
