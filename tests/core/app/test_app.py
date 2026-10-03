import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSystemTrayIcon
from tests.core.conftest import Env

from vector_embed.app import controller, hotkey
from vector_embed.app import main as app_main
from vector_embed.app.controller import Launcher, SearchOutcome, SearchService
from vector_embed.app.result_delegate import ROW_ROLE
from vector_embed.app.window import COMPACT_HEIGHT, EXPANDED_HEIGHT, Mode, SearchWindow
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.skills.search import SearchResult


def result(**kw: object) -> SearchResult:
    fields: dict[str, object] = {
        "path": r"D:\p\a.py",
        "project": "p",
        "kind": "code",
        "source": "",
        "symbol": "f",
        "start_line": 3,
        "end_line": 9,
        "page": 0,
        "snippet": "def f(): pass",
        "score": 0.5,
        "text": "def f(): pass",
    }
    fields.update(kw)
    return SearchResult(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- hotkey
def test_parse_hotkey() -> None:
    assert hotkey.parse_hotkey("ctrl+alt+space") == (0x2 | 0x1, 0x20)
    assert hotkey.parse_hotkey("Ctrl+Shift+K") == (0x2 | 0x4, ord("K"))
    assert hotkey.parse_hotkey("win+f9") == (0x8, 0x78)
    with pytest.raises(ValueError, match="non-modifier"):
        hotkey.parse_hotkey("ctrl+alt")
    with pytest.raises(ValueError, match="unknown key"):
        hotkey.parse_hotkey("ctrl+banana")


@pytest.mark.parametrize(
    ("spec", "key"),
    [
        ("ctrl+f1", 0x70),
        ("ctrl+f12", 0x7B),
        ("ctrl+f24", 0x87),
        ("alt+0", ord("0")),
        ("alt+z", 0x5A),
    ],
)
def test_function_and_plain_keys_map_to_their_virtual_keys(spec: str, key: int) -> None:
    assert hotkey.parse_hotkey(spec)[1] == key


@pytest.mark.parametrize("spec", ["ctrl+f0", "ctrl+f25", "ctrl+f99"])
def test_function_keys_outside_f1_to_f24_are_rejected(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown key"):
        hotkey.parse_hotkey(spec)


@pytest.mark.parametrize(
    ("spec", "key"),
    [
        ("ctrl+;", 0xBA),
        ("ctrl+=", 0xBB),
        ("ctrl+,", 0xBC),
        ("ctrl+-", 0xBD),
        ("ctrl+.", 0xBE),
        ("ctrl+/", 0xBF),
        ("ctrl+`", 0xC0),
        ("ctrl+[", 0xDB),
        ("ctrl+]", 0xDD),
        ("ctrl+'", 0xDE),
        ("ctrl+plus", 0xBB),
        ("ctrl+minus", 0xBD),
    ],
)
def test_punctuation_uses_the_oem_virtual_keys_not_ascii(spec: str, key: int) -> None:
    assert hotkey.parse_hotkey(spec)[1] == key  # ord(";") is 0x3B, which is the wrong key


def test_a_bare_key_would_steal_it_from_every_program_so_it_is_refused() -> None:
    with pytest.raises(ValueError, match="modifier"):
        hotkey.parse_hotkey("k")
    assert hotkey.parse_hotkey("f9") == (0, 0x78)  # function keys are fine on their own


def test_hotkey_registers_and_releases(qapp: QApplication) -> None:
    flt = hotkey.HotkeyFilter(lambda: None)
    if flt.register("ctrl+alt+shift+f12"):  # unlikely to collide with another program
        flt.unregister()
        assert not flt.registered
    flt.unregister()  # idempotent


def test_native_filter_dispatches_only_our_hotkey(qapp: QApplication) -> None:
    calls: list[int] = []
    flt = hotkey.HotkeyFilter(lambda: calls.append(1))

    def deliver(message: int, wparam: int) -> tuple[bool, int]:
        msg = wintypes.MSG()
        msg.message, msg.wParam = message, wparam
        return flt.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))

    flt._active_id = flt.HOTKEY_IDS[0]
    assert deliver(hotkey.WM_HOTKEY, flt.HOTKEY_IDS[0]) == (True, 0)
    assert deliver(hotkey.WM_HOTKEY, 1) == (False, 0)
    assert deliver(0x0001, flt.HOTKEY_IDS[0]) == (False, 0)
    assert flt.nativeEventFilter(b"other", 0) == (False, 0)
    assert calls == [1]


# ------------------------------------------------------------------- controller
def test_guess_project_from_editor_titles() -> None:
    assert controller.guess_project("a.py - my-proj - Visual Studio Code") == "my-proj"
    assert controller.guess_project("x - repo - Cursor") == "repo"
    assert controller.guess_project("Some random window") is None


def test_foreground_title_never_raises() -> None:
    assert isinstance(controller.foreground_title(), str)


def fake_stat(size: int) -> object:
    return lambda _path: os.stat_result((0, 0, 0, 0, 0, 0, size, 0, 0, 0))


def test_result_row_carries_what_the_list_draws() -> None:
    hit = result(extra_hits=2, kind="image", page=4, mtime=1_700_000_000)
    row = controller.result_row(hit, fake_stat(2048))  # type: ignore[arg-type]
    assert row.name == "a.py"
    assert row.detail == "·  f  (page 4)  +2 more"
    assert row.path == r"D:\p\a.py"
    assert (row.project, row.snippet, row.tag) == ("p", "def f(): pass", "IMG")
    assert row.size == "2.0 KB"
    assert row.modified == time.strftime("%Y-%m-%d", time.localtime(1_700_000_000))
    assert row.meta == f"{row.modified}  ·  2.0 KB"
    assert not row.is_image  # a page of a document, not a whole image


def test_result_row_for_a_bare_whole_image() -> None:
    row = controller.result_row(
        result(symbol="<module>", start_line=0, project="", kind="image"),
        fake_stat(5),  # type: ignore[arg-type]
    )
    assert row.detail == ""
    assert row.is_image
    assert row.size == "5 B"
    assert row.modified == ""
    assert row.meta == "5 B"


def test_result_row_survives_a_missing_file() -> None:
    def gone(_path: str) -> os.stat_result:
        raise FileNotFoundError

    row = controller.result_row(result(), gone)
    assert (row.size, row.meta) == ("", "")


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (5 * 1024**2, "5.0 MB"),
        (3 * 1024**3, "3.0 GB"),
    ],
)
def test_format_size(size: int, text: str) -> None:
    assert controller.format_size(size) == text


class TestLauncher:
    def make(self, which: str | None = "code") -> tuple[Launcher, list[object]]:
        calls: list[object] = []
        launcher = Launcher(
            startfile=lambda p: calls.append(("open", p)),
            popen=lambda cmd, **kw: calls.append(("run", cmd, kw)),
            which=lambda _name: which,
        )
        return launcher, calls

    def test_open_and_reveal(self) -> None:
        launcher, calls = self.make()
        launcher.open_file(result())
        launcher.reveal(result())
        assert calls[0] == ("open", r"D:\p\a.py")
        assert calls[1][1][:2] == ["explorer", "/select,"]  # type: ignore[index]

    def test_editor_jumps_to_the_line(self) -> None:
        launcher, calls = self.make()
        launcher.open_in_editor(result())
        assert calls[0][1] == ["code", "-g", r"D:\p\a.py:3"]  # type: ignore[index]
        launcher.open_in_editor(result(start_line=0))
        assert calls[1][1] == ["code", "-g", r"D:\p\a.py"]  # type: ignore[index]

    def test_without_vscode_falls_back_to_the_default_app(self) -> None:
        launcher, calls = self.make(which=None)
        launcher.open_in_editor(result())
        assert calls == [("open", r"D:\p\a.py")]


class TestSearchService:
    def test_search_and_warm(self, skill_ctx: SkillContext, env: Env) -> None:
        (env.root / "payments.py").write_text("def retry_payments():\n    return 1\n", "utf-8")
        env.indexer.index_paths([str(env.root / "payments.py")])
        builds: list[int] = []

        def factory() -> SkillContext:
            builds.append(1)
            return skill_ctx

        service = SearchService(factory)
        assert builds == []  # nothing is built until needed
        service.warm()
        outcome = service.search("retry payments", None)
        assert [Path(r.path).name for r in outcome.results] == ["payments.py"]
        assert outcome.message == ""
        assert builds == [1]
        assert service.on_battery() is False

    def test_errors_become_a_message(self, skill_ctx: SkillContext) -> None:
        skill_ctx.power._settings = skill_ctx.settings.power.model_copy(
            update={"search_on_battery": False}
        )
        skill_ctx.power._probe = lambda: False
        outcome = SearchService(lambda: skill_ctx).search("x", None)
        assert outcome.results == []
        assert "disabled on battery" in outcome.message


# ----------------------------------------------------------------------- window
class FakeService:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[tuple[str, str | None]] = []
        self.warmed = 0
        self.battery = False
        self.resets = 0
        self.released = 0

    def reset(self) -> None:
        self.resets += 1

    def release(self) -> None:
        self.released += 1

    def warm(self) -> None:
        self.warmed += 1

    def on_battery(self) -> bool:
        return self.battery

    def search(self, query: str, project: str | None) -> SearchOutcome:
        self.queries.append((query, project))
        return SearchOutcome(self.results, 12.0)


@pytest.fixture
def window(qapp: QApplication, tmp_path: Path) -> tuple[SearchWindow, FakeService, list[object]]:
    service = FakeService([result(path=r"D:\p\a.py"), result(path=r"D:\p\b.py", symbol="g")])
    calls: list[object] = []
    launcher = Launcher(
        startfile=lambda p: calls.append(("open", p)),
        popen=lambda cmd, **kw: calls.append(("run", cmd)),
        which=lambda _n: "code",
    )
    win = SearchWindow(service, launcher, tmp_path / "thumbs")  # type: ignore[arg-type]
    return win, service, calls


def wait_for(qapp: QApplication, condition, timeout: float = 5.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while not condition() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)


class TestWindow:
    def test_results_are_listed_with_row_data(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, service, _ = window
        win.show_results(SearchOutcome(service.results, 12.0))
        assert win.list.count() == 2
        assert "a.py" in win.list.item(0).text()
        row = win.list.item(0).data(ROW_ROLE)
        assert (row.name, row.path) == ("a.py", r"D:\p\a.py")
        assert win.status.text() == "2 results in 12 ms"

    def test_popup_is_just_the_search_bar_until_there_are_results(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, service, _ = window
        win.show()
        assert win.height() == COMPACT_HEIGHT
        assert win.body.isHidden()
        assert win.mode_label.isHidden()
        assert win.answer.isHidden()
        win.show_results(SearchOutcome(service.results, 1.0))
        assert not win.body.isHidden()
        assert win.height() == EXPANDED_HEIGHT
        win.input.setText("x")
        win.input.clear()
        win.run_search()
        assert win.body.isHidden()
        assert win.height() == COMPACT_HEIGHT

    def test_other_modes_show_their_label_and_answer_pane(
        self, qapp: QApplication, tmp_path: Path
    ) -> None:
        from tests.core.app.test_modes import FakeAssistant

        win = SearchWindow(
            FakeService(),  # type: ignore[arg-type]
            Launcher(),
            tmp_path,
            FakeAssistant(),  # type: ignore[arg-type]
        )
        win.show()
        win.set_mode(Mode.ASK)
        assert not win.mode_label.isHidden()
        assert not win.answer.isHidden()
        assert win.height() == EXPANDED_HEIGHT
        win.set_mode(Mode.SEARCH)
        assert win.mode_label.isHidden()
        assert win.answer.isHidden()

    def test_no_results_shows_the_message(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, _, _ = window
        win.show_results(SearchOutcome([], 1.0, "SearchDisabledError: nope"))
        assert win.status.text() == "SearchDisabledError: nope"
        win.show_results(SearchOutcome([], 1.0))
        assert win.status.text() == "No results."
        assert win.body.isHidden()

    def test_stale_generations_are_ignored(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, service, _ = window
        win._generation = 5
        win._on_outcome(4, SearchOutcome(service.results, 1.0))
        assert win.list.count() == 0
        win._on_outcome(5, SearchOutcome(service.results, 1.0))
        assert win.list.count() == 2

    def test_typing_runs_a_debounced_background_search(
        self, qapp: QApplication, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, service, _ = window
        win._project = "p"
        win.input.setText("retry payments")
        wait_for(qapp, lambda: win.list.count() == 2)
        assert service.queries == [("retry payments", "p")]
        win.input.setText("")
        win.run_search()
        assert win.list.count() == 0

    def test_keyboard_navigation_and_actions(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, service, calls = window
        win.show()
        win.show_results(SearchOutcome(service.results, 1.0))
        QTest.keyClick(win.input, Qt.Key.Key_Down)
        assert win.list.currentRow() == 1
        QTest.keyClick(win.input, Qt.Key.Key_Down)
        assert win.list.currentRow() == 1  # clamped
        QTest.keyClick(win.input, Qt.Key.Key_Up)
        assert win.list.currentRow() == 0
        QTest.keyClick(win.input, Qt.Key.Key_Return)
        QTest.keyClick(win.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
        QTest.keyClick(win.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
        kinds = [c[0] for c in calls]  # type: ignore[index]
        assert kinds == ["open", "run", "run"]
        assert not win.isVisible()  # actions hide the window
        assert calls[2][1][:2] == ["code", "-g"]  # type: ignore[index]
        QTest.keyClick(win.input, Qt.Key.Key_A)  # ordinary keys are left to the line edit

    def test_actions_do_nothing_without_a_selection(
        self, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, _, calls = window
        win.open_selected()
        win.reveal_selected()
        win.code_selected()
        assert calls == []
        assert win.selected() is None

    def test_summon_warms_up_and_reports_battery(
        self, qapp: QApplication, window: tuple[SearchWindow, FakeService, list[object]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:  # fmt: skip
        win, service, _ = window
        monkeypatch.setattr(controller, "foreground_title", lambda: "a.py - proj - Cursor")
        monkeypatch.setattr(
            "vector_embed.app.window.foreground_title", lambda: "a.py - proj - Cursor"
        )
        service.battery = True
        win.summon()
        wait_for(qapp, lambda: service.warmed == 1)
        assert service.warmed == 1
        assert "project: proj" in win.status.text()
        assert "on battery" in win.status.text()
        assert win.isVisible()

    def test_battery_probe_failure_is_cosmetic(
        self,
        window: tuple[SearchWindow, FakeService, list[object]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        win, service, _ = window

        def boom() -> bool:
            raise RuntimeError("no index yet")

        monkeypatch.setattr(service, "on_battery", boom)
        assert win._on_battery() is False

    def test_focus_loss_hides_the_window(
        self, qapp: QApplication, window: tuple[SearchWindow, FakeService, list[object]]
    ) -> None:
        win, _, _ = window
        win.show()
        win._hide_if_inactive()
        assert not win.isVisible() or win.isActiveWindow()


# ------------------------------------------------------------------------- main
def test_tray_icon_is_drawn(qapp: QApplication) -> None:
    assert not app_main.tray_icon().isNull()


def test_build_window_wires_the_service(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch, skill_ctx: SkillContext
) -> None:
    monkeypatch.setattr(app_main.runtime, "build_skill_context", lambda _s, _st: skill_ctx)
    win = app_main.build_window(env.settings, env.state)
    assert isinstance(win, SearchWindow)
    win._service.search("x", None)  # builds the context lazily through the factory


@pytest.mark.parametrize("hotkey_ok", [True, False])
def test_main_runs_the_event_loop(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch, hotkey_ok: bool
) -> None:
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "install_excepthooks", lambda: None)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: 7)
    monkeypatch.setattr(app_main.HotkeyFilter, "register", lambda _self, _spec: hotkey_ok)
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(QSystemTrayIcon, "showMessage", lambda *_a: None)
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: None)
    monkeypatch.setattr(app_main, "run_setup_wizard", lambda *_a, **_k: None)
    summoned: list[int] = []
    monkeypatch.setattr(app_main.SearchWindow, "summon", lambda _self: summoned.append(1))
    assert app_main.main(["--show"]) == 7
    assert summoned == [1]


def test_main_survives_an_invalid_hotkey(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = env.settings.model_copy(
        update={"search": env.settings.search.model_copy(update={"hotkey": "ctrl+banana"})}
    )
    monkeypatch.setattr(app_main, "load_settings", lambda: settings)
    monkeypatch.setattr(app_main, "install_excepthooks", lambda: None)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: 0)
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(QSystemTrayIcon, "showMessage", lambda *_a: None)
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: None)
    monkeypatch.setattr(app_main, "run_setup_wizard", lambda *_a, **_k: None)
    assert app_main.main([]) == 0


def test_main_reports_invalid_settings_instead_of_dying_silently(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[str] = []

    def broken() -> object:
        raise app_main.SettingsError("invalid settings in C:/x/settings.toml: bad key")

    monkeypatch.setattr(app_main, "load_settings", broken)
    monkeypatch.setattr(app_main, "install_excepthooks", lambda: None)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(app_main, "show_settings_error", shown.append)
    assert app_main.main([]) == app_main.EXIT_BAD_SETTINGS
    assert "settings.toml" in shown[0]


def test_main_exits_when_another_copy_is_running(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "install_excepthooks", lambda: None)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(app_main, "run_app", lambda *_a: pytest.fail("a second app started"))
    with app_main.single_instance("app", env.settings.storage.data_dir) as first:
        assert first
        assert app_main.main([]) == app_main.EXIT_OK


def test_the_battery_hint_never_builds_the_context_on_the_ui_thread() -> None:
    built: list[int] = []

    def factory() -> SkillContext:
        built.append(1)
        raise AssertionError("must not be built from the UI thread")

    service = controller.SearchService(factory)
    assert service.on_battery() is False
    assert built == []


class FakeUser32:
    """Records hotkey registrations; ``taken`` holds the (modifiers, key) other programs own."""

    def __init__(self) -> None:
        self.live: dict[int, tuple[int, int]] = {}
        self.taken: set[tuple[int, int]] = set()

    def RegisterHotKey(self, _hwnd: object, ident: int, modifiers: int, key: int) -> bool:  # noqa: N802
        combo = (modifiers & ~hotkey.MOD_NOREPEAT, key)
        if combo in self.taken or ident in self.live:
            return False
        self.live[ident] = combo
        return True

    def UnregisterHotKey(self, _hwnd: object, ident: int) -> bool:  # noqa: N802
        return self.live.pop(ident, None) is not None


@pytest.fixture
def user32(monkeypatch: pytest.MonkeyPatch) -> FakeUser32:
    from types import SimpleNamespace

    fake = FakeUser32()
    monkeypatch.setattr(hotkey, "ctypes", SimpleNamespace(windll=SimpleNamespace(user32=fake)))
    return fake


class TestChangingTheHotkey:
    def test_the_old_key_is_released_only_after_the_new_one_is_secured(
        self, qapp: QApplication, user32: FakeUser32
    ) -> None:
        flt = hotkey.HotkeyFilter(lambda: None)
        assert flt.register("ctrl+alt+space")
        first = dict(user32.live)
        assert flt.register("ctrl+alt+f9")
        assert len(user32.live) == 1  # exactly one live hotkey: the new one
        assert user32.live != first
        assert flt.spec == "ctrl+alt+f9"
        assert flt.registered

    def test_a_refused_new_key_keeps_the_old_one_working(
        self, qapp: QApplication, user32: FakeUser32
    ) -> None:
        flt = hotkey.HotkeyFilter(lambda: None)
        assert flt.register("ctrl+alt+space")
        before = dict(user32.live)
        user32.taken.add(hotkey.parse_hotkey("ctrl+alt+f9"))  # another program owns it
        assert flt.register("ctrl+alt+f9") is False
        assert user32.live == before  # nothing was lost
        assert flt.spec == "ctrl+alt+space"
        assert flt.registered

    def test_an_invalid_spec_changes_nothing(self, qapp: QApplication, user32: FakeUser32) -> None:
        flt = hotkey.HotkeyFilter(lambda: None)
        flt.register("ctrl+alt+space")
        before = dict(user32.live)
        with pytest.raises(ValueError, match="unknown key"):
            flt.register("ctrl+banana")
        assert user32.live == before
        assert flt.spec == "ctrl+alt+space"

    def test_unregister_clears_everything_and_the_filter_only_answers_the_live_key(
        self, qapp: QApplication, user32: FakeUser32
    ) -> None:
        calls: list[int] = []
        flt = hotkey.HotkeyFilter(lambda: calls.append(1))
        flt.register("ctrl+alt+space")
        flt.register("ctrl+alt+f9")
        stale = next(i for i in flt.HOTKEY_IDS if i != flt._active_id)

        def deliver(ident: int) -> tuple[bool, int]:
            msg = wintypes.MSG()
            msg.message, msg.wParam = hotkey.WM_HOTKEY, ident
            return flt.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(msg))

        assert deliver(stale) == (False, 0)  # the released key's message is ignored
        assert deliver(flt._active_id) == (True, 0)  # type: ignore[arg-type]
        flt.unregister()
        assert user32.live == {}
        assert flt.spec is None
        assert not flt.registered

    def test_the_applier_reports_a_refused_key_without_losing_the_old_one(
        self, qapp: QApplication, user32: FakeUser32
    ) -> None:
        flt = hotkey.HotkeyFilter(lambda: None)
        flt.register("ctrl+alt+space")
        tips: list[str] = []

        class Tray:
            def setToolTip(self, text: str) -> None:  # noqa: N802
                tips.append(text)

        apply = app_main.hotkey_applier(flt, Tray())  # type: ignore[arg-type]
        apply("ctrl+alt+f8")
        assert tips[-1] == "Vector Embed (ctrl+alt+f8)"
        user32.taken.add(hotkey.parse_hotkey("ctrl+alt+f7"))
        apply("ctrl+alt+f7")
        assert tips[-1] == "Vector Embed (ctrl+alt+f8) - ctrl+alt+f7 is unavailable"
        apply("ctrl+nonsense")
        assert "unavailable" in tips[-1]
        assert flt.spec == "ctrl+alt+f8"


def test_quitting_from_the_tray_unloads_the_models(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutdowns: list[int] = []
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "install_excepthooks", lambda: None)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    # quitting makes ``exec`` return, and ``aboutToQuit`` fires inside it, however the app ends:
    # the tray's Quit, a log-off, Delete my data
    monkeypatch.setattr(qapp, "exec", lambda: (qapp.aboutToQuit.emit(), 0)[1])
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(QSystemTrayIcon, "showMessage", lambda *_a: None)
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: None)
    monkeypatch.setattr(app_main, "run_setup_wizard", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main.SearchWindow, "shutdown", lambda _self: shutdowns.append(1))
    assert app_main.main([]) == 0
    assert shutdowns == [1]
    qapp.aboutToQuit.emit()  # the connection was dropped with the window: no second call
    assert shutdowns == [1]
