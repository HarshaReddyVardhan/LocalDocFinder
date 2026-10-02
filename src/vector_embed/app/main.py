"""Entry point: tray icon, global hotkey and the search window.

pythonw -m vector_embed.app          (or ``python -m vector_embed.app --show``)
"""

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QSystemTrayIcon

from vector_embed.app.assistant import AssistantService
from vector_embed.app.controller import Launcher, SearchService
from vector_embed.app.hotkey import HotkeyFilter
from vector_embed.app.match_controller import MatchController
from vector_embed.app.models_controller import ModelsController
from vector_embed.app.settings_controller import SettingsController
from vector_embed.app.settings_window import SettingsWindow
from vector_embed.app.setup_controller import SetupController
from vector_embed.app.setup_wizard import SetupWizard
from vector_embed.app.update_scheduler import UpdateScheduler
from vector_embed.app.window import SearchWindow
from vector_embed.core import runtime
from vector_embed.core.autostart import Autostart
from vector_embed.core.idle import SystemActivity
from vector_embed.core.lifecycle import start_watcher
from vector_embed.core.logging_setup import configure_logging, install_excepthooks
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import probe_hardware
from vector_embed.core.process import is_frozen, single_instance
from vector_embed.core.secrets import KeyringStore
from vector_embed.core.settings import (
    SETTINGS_FILENAME,
    Settings,
    SettingsError,
    default_data_dir,
    load_settings,
)
from vector_embed.core.setup.flow import SETUP_COMPLETED_KEY
from vector_embed.core.setup.wiring import FlowBuilder, build_flow
from vector_embed.core.skills.base import SkillContext
from vector_embed.core.store.sqlite import StateDb
from vector_embed.core.updates import Updater, resolve_source

logger = logging.getLogger("app")

EXIT_OK = 0
EXIT_BAD_SETTINGS = 2


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
        pick_file=pick_document,
    )


def make_settings_controller(
    path: Path, state: StateDb, updater: Updater | None = None
) -> SettingsController:
    return SettingsController(
        path, state, KeyringStore(), apply_autostart=Autostart().apply, updater=updater
    )


def auto_check_enabled(path: Path) -> Callable[[], bool]:
    """Re-reads the setting each time, so toggling it in Settings takes effect immediately."""

    def enabled() -> bool:
        try:
            return load_settings(path).updates.auto_check
        except SettingsError:
            return True

    return enabled


def build_settings_window(
    settings: Settings,
    state: StateDb,
    context: Callable[[], SkillContext],
    on_hotkey: Callable[[str], None],
    updater: Updater | None = None,
) -> SettingsWindow:
    path = settings.storage.data_dir / SETTINGS_FILENAME
    window = SettingsWindow(
        make_settings_controller(path, state, updater), ModelsController(context, path)
    )
    window.hotkey_changed.connect(on_hotkey)
    return window


def announce_update(tray: QSystemTrayIcon, restart_action: QAction, version: str) -> None:
    restart_action.setVisible(True)
    tray.showMessage(
        "Vector Embed",
        f"Version {version} is ready. Choose Restart to update in the tray menu.",
        QSystemTrayIcon.MessageIcon.Information,
        8000,
    )


def setup_needed(state: StateDb) -> bool:
    return state.get_meta(SETUP_COMPLETED_KEY) is None


def run_setup_wizard(
    settings: Settings,
    state: StateDb,
    build: FlowBuilder = build_flow,
    updater: Updater | None = None,
) -> None:
    """Show the first-run wizard (also reachable from the tray); returns when it closes."""
    path = settings.storage.data_dir / SETTINGS_FILENAME
    wizard = SetupWizard(
        SetupController(build, settings, state),
        make_settings_controller(path, state, updater),
        load_catalog(settings.storage.data_dir),
        probe_hardware(),
    )
    wizard.exec()


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


def show_settings_error(message: str) -> None:
    QMessageBox.critical(None, "Vector Embed", message)


def load_settings_or_report(show: Callable[[str], None] | None = None) -> Settings | None:
    """The settings, or ``None`` after telling the user why the app cannot start."""
    try:
        return load_settings()
    except SettingsError as exc:
        logger.error("settings are invalid: %s", exc)
        (show or show_settings_error)(
            f"Vector Embed cannot start because its settings are invalid.\n\n{exc}"
        )
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="show the window immediately")
    parser.add_argument("--setup", action="store_true", help="run the setup wizard now")
    args = parser.parse_args(argv)
    install_excepthooks()
    # Logging first, so even a settings failure leaves a trace; reconfigured once settings load.
    configure_logging("app", default_data_dir() / runtime.LOGS_DIRNAME)
    app = QApplication(sys.argv[:1])
    app.setQuitOnLastWindowClosed(False)
    settings = load_settings_or_report()
    if settings is None:
        return EXIT_BAD_SETTINGS
    configure_logging("app", runtime.log_dir(settings), settings.log_level)
    with single_instance("app", settings.storage.data_dir) as acquired:
        if not acquired:
            logger.info("another copy of the app is already running")
            return EXIT_OK
        return run_app(app, settings, args)


def run_app(app: QApplication, settings: Settings, args: argparse.Namespace) -> int:
    with StateDb(settings.storage.data_dir) as state:
        context = make_context_factory(settings, state)
        updater = Updater(resolve_source(settings.updates.repo_url), state=state)
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
                    build_settings_window(
                        settings, state, context, hotkey_applier(hotkey, tray), updater
                    )
                )
            settings_window[0].open()

        menu.addAction("Search", window.summon)
        menu.addAction("Settings…", open_settings)
        menu.addAction(
            "Run setup again…", lambda: run_setup_wizard(settings, state, updater=updater)
        )
        restart_action = menu.addAction("Restart to update", updater.restart_to_update)
        restart_action.setVisible(False)  # shown once an update has been downloaded
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
        scheduler = UpdateScheduler(
            updater, auto_check_enabled(settings.storage.data_dir / SETTINGS_FILENAME)
        )
        scheduler.ready.connect(lambda version: announce_update(tray, restart_action, version))
        scheduler.start()
        if is_frozen():
            start_watcher()  # a fresh install has had no logon yet; a no-op if one is running
        if args.setup or setup_needed(state):
            run_setup_wizard(settings, state, updater=updater)
        if args.show:
            window.summon()
        code = app.exec()
        hotkey.unregister()
    return code


if __name__ == "__main__":
    sys.exit(main())
