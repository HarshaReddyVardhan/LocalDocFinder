"""Entry point: tray icon, global hotkey and the search window.

pythonw -m vector_embed.app          (or ``python -m vector_embed.app --show``)
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QSystemTrayIcon

from vector_embed.app.assistant import AssistantService
from vector_embed.app.controller import Launcher, SearchService
from vector_embed.app.hotkey import HotkeyFilter
from vector_embed.app.match_controller import MatchController
from vector_embed.app.models_controller import ModelsController
from vector_embed.app.settings_controller import SettingsController
from vector_embed.app.settings_window import SettingsWindow
from vector_embed.app.window import SearchWindow
from vector_embed.core import runtime
from vector_embed.core.idle import SystemActivity
from vector_embed.core.logging_setup import configure_logging
from vector_embed.core.secrets import KeyringStore
from vector_embed.core.settings import SETTINGS_FILENAME, Settings, load_settings
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.store.sqlite import StateDb

logger = logging.getLogger("app")


def tray_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#4c7dff"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, 56, 56)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "S")
    painter.end()
    return QIcon(pixmap)


def pick_document() -> str | None:
    """File dialog for ``+ Add file…`` in the Match panel."""
    path, _ = QFileDialog.getOpenFileName(None, "Add a document to match")
    return path or None


def make_context_factory(settings: Settings, state: StateDb) -> Callable[[], SkillContext]:
    """One lazily built skill context, shared by search, the assistant and the Settings window."""
    cache: list[SkillContext] = []
    activity = SystemActivity()

    def context() -> SkillContext:
        if not cache:
            cache.append(
                runtime.build_skill_context(
                    settings, state, fullscreen=activity.fullscreen_app_active
                )
            )
        return cache[0]

    return context


def build_window(
    settings: Settings, state: StateDb, context: Callable[[], SkillContext] | None = None
) -> SearchWindow:
    context = context or make_context_factory(settings, state)
    return SearchWindow(
        SearchService(context),
        Launcher(),
        settings.storage.data_dir / runtime.THUMBS_DIRNAME,
        AssistantService(context),
        matcher=MatchController(context),
        models=ModelsController(context, settings.storage.data_dir / SETTINGS_FILENAME),
        pick_file=pick_document,
    )


def build_settings_window(
    settings: Settings,
    state: StateDb,
    context: Callable[[], SkillContext],
    on_hotkey: Callable[[str], None],
) -> SettingsWindow:
    path = settings.storage.data_dir / SETTINGS_FILENAME
    window = SettingsWindow(
        SettingsController(path, state, KeyringStore()), ModelsController(context, path)
    )
    window.hotkey_changed.connect(on_hotkey)
    return window


def hotkey_applier(hotkey: HotkeyFilter, tray: QSystemTrayIcon) -> Callable[[str], None]:
    """Re-register the global hotkey right away when the user changes it in Settings."""

    def apply(spec: str) -> None:
        hotkey.unregister()
        try:
            ok = hotkey.register(spec)
        except ValueError:
            ok = False
        tray.setToolTip(f"Vector Embed ({spec})" + ("" if ok else " - hotkey unavailable"))

    return apply


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show the window immediately")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging("app", runtime.log_dir(settings), settings.log_level)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    with StateDb(settings.storage.data_dir) as state:
        context = make_context_factory(settings, state)
        window = build_window(settings, state, context)
        hotkey = HotkeyFilter(window.summon)
        app.installNativeEventFilter(hotkey)
        registered = False
        try:
            registered = hotkey.register(settings.search.hotkey)
        except ValueError:
            logger.exception("invalid hotkey %r", settings.search.hotkey)

        tray = QSystemTrayIcon(tray_icon(), app)
        menu = QMenu()
        settings_window: list[SettingsWindow] = []  # built on first use: idle cost stays near zero

        def open_settings() -> None:
            if not settings_window:
                settings_window.append(
                    build_settings_window(settings, state, context, hotkey_applier(hotkey, tray))
                )
            settings_window[0].open()

        menu.addAction("Search", window.summon)
        menu.addAction("Settings…", open_settings)
        menu.addAction("Quit", app.quit)
        tray.setContextMenu(menu)
        suffix = "" if registered else " - hotkey unavailable"
        tray.setToolTip(f"Vector Embed ({settings.search.hotkey}){suffix}")
        tray.activated.connect(
            lambda reason: (
                window.summon() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
            )
        )
        tray.show()
        if not registered:
            tray.showMessage(
                "Vector Embed",
                f"Could not register {settings.search.hotkey}; use the tray icon.",
                QSystemTrayIcon.MessageIcon.Warning,
                5000,
            )
        if args.show:
            window.summon()
        code = app.exec()
        hotkey.unregister()
    return code


if __name__ == "__main__":
    sys.exit(main())
