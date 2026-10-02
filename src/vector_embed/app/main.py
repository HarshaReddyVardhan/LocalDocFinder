"""Entry point: tray icon, global hotkey and the search window.

pythonw -m vector_embed.app          (or ``python -m vector_embed.app --show``)
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QSystemTrayIcon

from vector_embed.app.assistant import AssistantService
from vector_embed.app.controller import Launcher, SearchService
from vector_embed.app.hotkey import HotkeyFilter
from vector_embed.app.match_controller import MatchController
from vector_embed.app.window import SearchWindow
from vector_embed.core import runtime
from vector_embed.core.idle import SystemActivity
from vector_embed.core.logging_setup import configure_logging
from vector_embed.core.settings import Settings, load_settings
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


def build_window(settings: Settings, state: StateDb) -> SearchWindow:
    """One lazily built skill context is shared by search and the assistant."""
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

    return SearchWindow(
        SearchService(context),
        Launcher(),
        settings.storage.data_dir / runtime.THUMBS_DIRNAME,
        AssistantService(context),
        matcher=MatchController(context),
        pick_file=pick_document,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show the window immediately")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging("app", runtime.log_dir(settings), settings.log_level)

    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    with StateDb(settings.storage.data_dir) as state:
        window = build_window(settings, state)
        hotkey = HotkeyFilter(window.summon)
        app.installNativeEventFilter(hotkey)
        registered = False
        try:
            registered = hotkey.register(settings.search.hotkey)
        except ValueError:
            logger.exception("invalid hotkey %r", settings.search.hotkey)

        tray = QSystemTrayIcon(tray_icon(), app)
        menu = QMenu()
        menu.addAction("Search", window.summon)
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
