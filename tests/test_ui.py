import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ui.app import HotkeyFilter, guess_project, parse_hotkey


def test_parse_hotkey():
    assert parse_hotkey("ctrl+alt+space") == (0x2 | 0x1, 0x20)
    assert parse_hotkey("Ctrl+Shift+K") == (0x2 | 0x4, ord("K"))
    assert parse_hotkey("win+f9") == (0x8, 0x78)
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+alt")
    with pytest.raises(ValueError):
        parse_hotkey("ctrl+banana")


def test_guess_project():
    assert guess_project("a.py - my-proj - Visual Studio Code") == "my-proj"
    assert guess_project("Some random window") is None


def test_hotkey_can_register_and_release():
    hk = HotkeyFilter(lambda: None)
    ok = hk.register("ctrl+alt+shift+f12")  # unlikely to collide with anything
    if ok:
        hk.unregister()
        assert not hk.registered


def test_window_renders_results(tmp_path, monkeypatch):
    from ui.app import SearchWindow
    from search import Result
    QApplication.instance() or QApplication([])
    win = SearchWindow(tmp_path)
    r = Result(path=r"D:\p\a.py", project="p", kind="code", source="", symbol="f", start_line=3,
               end_line=9, page=0, snippet="def f(): pass", score=0.5, text="def f(): pass")
    win._gen = 1
    win._on_results(1, [r], 12.0, "")
    assert win.list.count() == 1
    assert "a.py" in win.list.item(0).text()
    assert win.preview.toPlainText() == "def f(): pass"
    win._on_results(0, [], 1.0, "stale")  # stale generation ignored
    assert win.list.count() == 1
