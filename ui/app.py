"""Search window: global hotkey -> frameless popup -> hybrid search -> open / reveal / VS Code.

    pythonw -m ui.app          (or: python -m ui.app --data-dir D:\\somewhere)

Keys:  Enter open   Ctrl+Enter reveal in Explorer   Shift+Enter open in VS Code at the line   Esc hide
"""
import ctypes
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMenu, QPlainTextEdit, QSplitter, QStackedWidget, QSystemTrayIcon,
                               QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import indexer_config as cfg  # noqa: E402
import power  # noqa: E402

WM_HOTKEY = 0x0312
MOD = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
VK = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B}


def parse_hotkey(spec: str):
    """'ctrl+alt+space' -> (modifiers, virtual key)."""
    mods, vk = 0, None
    for part in spec.lower().replace(" ", "").split("+"):
        if part in MOD:
            mods |= MOD[part]
        elif part in VK:
            vk = VK[part]
        elif len(part) == 1:
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            vk = 0x70 + int(part[1:]) - 1
        else:
            raise ValueError(f"unknown key in hotkey: {part}")
    if vk is None:
        raise ValueError("hotkey needs a non-modifier key")
    return mods, vk


class HotkeyFilter(QAbstractNativeEventFilter):
    HOTKEY_ID = 0x5645  # 'VE'

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.registered = False

    def register(self, spec: str) -> bool:
        mods, vk = parse_hotkey(spec)
        MOD_NOREPEAT = 0x4000
        self.registered = bool(ctypes.windll.user32.RegisterHotKey(None, self.HOTKEY_ID, mods | MOD_NOREPEAT, vk))
        return self.registered

    def unregister(self) -> None:
        if self.registered:
            ctypes.windll.user32.UnregisterHotKey(None, self.HOTKEY_ID)
            self.registered = False

    def nativeEventFilter(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam == self.HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0


def foreground_title() -> str:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value
    except Exception:
        return ""


def guess_project(title: str) -> Optional[str]:
    """VS Code / Cursor titles look like 'file.py - project - Visual Studio Code'."""
    parts = [p.strip() for p in title.split(" - ")]
    if len(parts) >= 3 and parts[-1] in ("Visual Studio Code", "Cursor", "Windsurf"):
        return parts[-2]
    return None


class _Signals(QObject):
    done = Signal(int, list, float, str)


class _SearchJob(QRunnable):
    def __init__(self, gen: int, query: str, searcher_getter, project: Optional[str], signals: _Signals):
        super().__init__()
        self.gen, self.query, self.get, self.project, self.signals = gen, query, searcher_getter, project, signals

    def run(self):
        t = time.time()
        try:
            results = self.get().search(self.query, current_project=self.project)
            msg = ""
        except Exception as e:  # SearchDisabled, Ollama down, empty index ...
            results, msg = [], f"{type(e).__name__}: {e}"
        self.signals.done.emit(self.gen, results, (time.time() - t) * 1000, msg)


class _WarmJob(QRunnable):
    def __init__(self, searcher_getter):
        super().__init__()
        self.get = searcher_getter

    def run(self):
        try:
            self.get().warm()
        except Exception:
            pass


class SearchWindow(QWidget):
    def __init__(self, data_dir: Path = None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.data_dir = Path(data_dir or cfg.DATA_DIR)
        self._searcher = None
        self._gen = 0
        self._results: List = []
        self._project: Optional[str] = None
        self.pool = QThreadPool.globalInstance()
        self.signals = _Signals()
        self.signals.done.connect(self._on_results)

        self.setWindowTitle("Vector Embed")
        self.resize(1000, 560)
        self.setStyleSheet("""
            QWidget { background:#1e1f24; color:#e6e6e6; font-size:13px; }
            QLineEdit { background:#2a2c33; border:1px solid #3b3e47; border-radius:6px; padding:9px 12px; font-size:16px; }
            QListWidget { background:#1e1f24; border:none; outline:0; }
            QListWidget::item { padding:6px 8px; border-bottom:1px solid #2a2c33; }
            QListWidget::item:selected { background:#33405a; }
            QPlainTextEdit { background:#17181c; border:1px solid #2a2c33; font-family:Consolas; font-size:12px; }
            QLabel#status { color:#8a8f9c; padding:2px 6px; }
        """)
        self.input = QLineEdit(placeholderText="Search code, notes, PDFs, images…   type:code  proj:name  ext:py  in:D:\\x  after:2026-01")
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.preview = QPlainTextEdit(readOnly=True)
        self.image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.pane = QStackedWidget()
        self.pane.addWidget(self.preview)
        self.pane.addWidget(self.image)
        self.status = QLabel("", objectName="status")

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(self.pane)
        split.setSizes([520, 480])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.addWidget(self.input)
        lay.addWidget(split, 1)
        lay.addWidget(self.status)

        self.timer = QTimer(self, singleShot=True, interval=180)
        self.timer.timeout.connect(self._run_search)
        self.input.textChanged.connect(lambda _: self.timer.start())
        self.list.currentRowChanged.connect(self._show_preview)
        self.list.itemActivated.connect(lambda _: self.open_selected())
        self.input.installEventFilter(self)
        QShortcut(QKeySequence("Esc"), self, activated=self.hide)

    # ---------------------------------------------------------------- searcher
    def _get_searcher(self):
        if self._searcher is None:
            from embedder import Embedder
            from search import Searcher
            from store import Store
            emb = Embedder()
            self._searcher = Searcher(Store(self.data_dir, emb.model, dim=None, read_only=True), emb)
        return self._searcher

    def summon(self) -> None:
        self._project = guess_project(foreground_title())
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()
        self.input.selectAll()
        self.pool.start(_WarmJob(self._get_searcher))  # hides the 1-2 s model load while typing
        mode = "" if power.on_ac_power() else "  ·  on battery: searching on CPU"
        self.status.setText((f"project: {self._project}" if self._project else "") + mode)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type().name == "ActivationChange" and not self.isActiveWindow() and self.isVisible():
            QTimer.singleShot(150, lambda: None if self.isActiveWindow() else self.hide())

    # ------------------------------------------------------------------ search
    def _run_search(self) -> None:
        q = self.input.text().strip()
        self._gen += 1
        if not q:
            self.list.clear()
            self.preview.clear()
            return
        self.pool.start(_SearchJob(self._gen, q, self._get_searcher, self._project, self.signals))

    def _on_results(self, gen: int, results: list, ms: float, msg: str) -> None:
        if gen != self._gen:
            return  # stale
        self._results = results
        self.list.clear()
        for r in results:
            title = os.path.basename(r.path) + (f"  ·  {r.symbol}" if r.symbol and r.symbol != "<module>" else "")
            loc = f"  ({r.location})" if r.location else ""
            more = f"  +{r.extra_hits} more" if r.extra_hits else ""
            tag = {"image": "IMG", "ai-note": "NOTE", "outline": "FILE"}.get(r.kind, r.kind.upper())
            item = QListWidgetItem(f"[{tag}] {title}{loc}{more}\n{r.project + '  ' if r.project else ''}{r.path}\n{r.snippet[:110]}")
            self.list.addItem(item)
        if results:
            self.list.setCurrentRow(0)
        else:
            self.preview.setPlainText(msg or "No results.")
            self.pane.setCurrentIndex(0)
        self.status.setText(msg or f"{len(results)} results in {ms:.0f} ms")

    def _show_preview(self, row: int) -> None:
        if not 0 <= row < len(self._results):
            return
        r = self._results[row]
        if r.kind == "image" and not r.page:
            from extract.image import thumbnail_path
            import xxhash
            try:
                digest = xxhash.xxh3_64_hexdigest(Path(r.path).read_bytes())
                pm = QPixmap(str(thumbnail_path(digest)))
                if not pm.isNull():
                    self.image.setPixmap(pm)
                    self.pane.setCurrentIndex(1)
                    return
            except OSError:
                pass
        self.preview.setPlainText(r.text)
        self.pane.setCurrentIndex(0)

    # ----------------------------------------------------------------- actions
    def _selected(self):
        row = self.list.currentRow()
        return self._results[row] if 0 <= row < len(self._results) else None

    def open_selected(self) -> None:
        r = self._selected()
        if r:
            self.hide()
            os.startfile(r.path)

    def reveal_selected(self) -> None:
        r = self._selected()
        if r:
            self.hide()
            subprocess.Popen(["explorer", "/select,", os.path.normpath(r.path)])

    def code_selected(self) -> None:
        r = self._selected()
        if not r:
            return
        self.hide()
        code = shutil.which("code")
        target = f"{r.path}:{r.start_line}" if r.start_line else r.path
        if code:
            subprocess.Popen([code, "-g", target], creationflags=0x08000000)
        else:
            os.startfile(r.path)

    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type().name == "KeyPress":
            key, mods = ev.key(), ev.modifiers()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                row = self.list.currentRow() + (1 if key == Qt.Key.Key_Down else -1)
                self.list.setCurrentRow(max(0, min(self.list.count() - 1, row)))
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if mods & Qt.KeyboardModifier.ControlModifier:
                    self.reveal_selected()
                elif mods & Qt.KeyboardModifier.ShiftModifier:
                    self.code_selected()
                else:
                    self.open_selected()
                return True
        return super().eventFilter(obj, ev)


def _tray_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#4c7dff"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setPen(QColor("white"))
    f = QFont("Segoe UI", 28, QFont.Weight.Bold)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    p.end()
    return QIcon(pm)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir")
    ap.add_argument("--show", action="store_true", help="show the window immediately (testing)")
    a = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    win = SearchWindow(a.data_dir)

    hk = HotkeyFilter(win.summon)
    app.installNativeEventFilter(hk)
    ok = False
    try:
        ok = hk.register(cfg.HOTKEY)
    except ValueError as e:
        print(f"bad HOTKEY: {e}", file=sys.stderr)

    tray = QSystemTrayIcon(_tray_icon(), app)
    menu = QMenu()
    menu.addAction("Search", win.summon)
    menu.addAction("Quit", app.quit)
    tray.setContextMenu(menu)
    tray.setToolTip(f"Vector Embed search ({cfg.HOTKEY})" + ("" if ok else " - hotkey unavailable"))
    tray.activated.connect(lambda reason: win.summon() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
    tray.show()
    if not ok:
        tray.showMessage("Vector Embed", f"Could not register {cfg.HOTKEY}; use the tray icon.",
                         QSystemTrayIcon.MessageIcon.Warning, 5000)
    if a.show:
        win.summon()
    code = app.exec()
    hk.unregister()
    return code


if __name__ == "__main__":
    sys.exit(main())
