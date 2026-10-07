"""Entry point: tray icon, global hotkey and the search window.

pythonw -m localdoc_finder.app          (or ``python -m localdoc_finder.app --show``)
"""

import argparse
import ctypes
import logging
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Generic, TypeVar

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu, QMessageBox, QSystemTrayIcon

from localdoc_finder.app.assistant import AssistantService
from localdoc_finder.app.cloud_switcher import CloudSwitcher
from localdoc_finder.app.controller import Launcher, SearchService
from localdoc_finder.app.hotkey import HotkeyFilter
from localdoc_finder.app.match_controller import MatchController
from localdoc_finder.app.models_controller import ModelsController
from localdoc_finder.app.settings_controller import SettingsController
from localdoc_finder.app.settings_window import SettingsWindow
from localdoc_finder.app.setup_controller import SetupController
from localdoc_finder.app.setup_wizard import SetupWizard
from localdoc_finder.app.theme import apply_theme
from localdoc_finder.app.update_scheduler import UpdateScheduler
from localdoc_finder.app.window import SearchWindow
from localdoc_finder.core import runtime
from localdoc_finder.core.autostart import Autostart
from localdoc_finder.core.cloud import CloudConsent
from localdoc_finder.core.features import enabled_features
from localdoc_finder.core.idle import SystemActivity
from localdoc_finder.core.indexing_control import IndexingControl
from localdoc_finder.core.lifecycle import start_watcher
from localdoc_finder.core.logging_setup import configure_logging, install_excepthooks
from localdoc_finder.core.models.catalog import load_catalog
from localdoc_finder.core.models.hardware import on_ac_power, probe_hardware
from localdoc_finder.core.ollama_service import ensure_ollama_running
from localdoc_finder.core.process import is_frozen, single_instance
from localdoc_finder.core.secrets import KeyringStore
from localdoc_finder.core.settings import (
    Settings,
    SettingsError,
    default_data_dir,
    load_settings,
)
from localdoc_finder.core.setup.flow import SETUP_COMPLETED_KEY
from localdoc_finder.core.setup.ollama_install import OllamaState
from localdoc_finder.core.setup.wiring import FlowBuilder, build_flow
from localdoc_finder.core.skills.base import SkillContext
from localdoc_finder.core.store.sqlite import StateDb
from localdoc_finder.core.terms import accept_terms, terms_accepted
from localdoc_finder.core.updates import Updater, resolve_source

logger = logging.getLogger("app")
T = TypeVar("T")

_open_wizards: list[SetupWizard] = []  # at most one: see run_setup_wizard

EXIT_OK = 0
EXIT_BAD_SETTINGS = 2


TRAY_ICON_FILE = Path(__file__).with_name("assets") / "icon.png"  # drawn by packaging/make_icon.py
APP_USER_MODEL_ID = "LocalDocFinder.App"


def tray_icon() -> QIcon:
    return QIcon(str(TRAY_ICON_FILE))


def _set_app_user_model_id(app_id: str) -> None:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore[attr-defined,unused-ignore]


def use_app_icon(
    app: QApplication, set_app_id: Callable[[str], None] = _set_app_user_model_id
) -> None:
    """The tray's icon on every window and its taskbar button.

    Run from source, Windows groups the windows under python.exe and shows Python's icon; an
    app id of our own gives them their own taskbar button. Must run before any window exists.
    """
    app.setWindowIcon(tray_icon())
    if not is_frozen():  # the installed exe is its own taskbar identity already
        set_app_id(APP_USER_MODEL_ID)


def pick_document() -> str | None:
    """File dialog for ``+ Add file…`` in the Match panel."""
    path, _ = QFileDialog.getOpenFileName(None, "Add a document to match")
    return path or None


class ContextFactory:
    """One lazily built skill context shared by search, the assistant and the Settings window.

    ``invalidate`` drops it after a settings change; the next call re-reads the settings file, so
    changes (privacy, cloud, models...) apply without restarting the app.
    """

    def __init__(
        self,
        settings: Settings,
        state: StateDb,
        reload: Callable[[], Settings] | None = None,
        ensure_server: Callable[[str], object] = lambda _host: None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._reload = reload
        self._ensure_server = ensure_server
        self._cache: SkillContext | None = None
        self._lock = threading.Lock()  # built on a worker thread; two must not race
        self._activity = SystemActivity()
        # One for the whole run, so a "don't ask again" outlives context rebuilds; a restart
        # starts with none.
        self.consent = CloudConsent()

    def __call__(self) -> SkillContext:
        with self._lock:
            if self._cache is None:
                self._ensure_server(self._settings.ollama_host)  # start Ollama if it is stopped
                self._cache = runtime.build_skill_context(
                    self._settings,
                    self._state,
                    fullscreen=self._activity.fullscreen_app_active,
                    consent=self.consent,
                )
            return self._cache

    @property
    def settings(self) -> Settings:
        """The settings in force; replaced by ``invalidate`` when the file changes."""
        return self._settings

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None
            if self._reload is not None:
                try:
                    self._settings = self._reload()
                except SettingsError:
                    logger.warning("settings changed but are now invalid; keeping the old ones")


def make_context_factory(
    settings: Settings, state: StateDb, reload: Callable[[], Settings] | None = None
) -> ContextFactory:
    return ContextFactory(settings, state, reload, ensure_ollama_running)


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
        features=lambda: enabled_features(
            context.settings if isinstance(context, ContextFactory) else settings
        ),
    )


def make_settings_controller(
    path: Path,
    state: StateDb,
    updater: Updater | None = None,
    on_changed: Callable[[], None] = lambda: None,
    forget_consent: Callable[[], None] = lambda: None,
) -> SettingsController:
    return SettingsController(
        path,
        state,
        KeyringStore(),
        apply_autostart=Autostart().apply,
        updater=updater,
        on_changed=on_changed,
        forget_consent=forget_consent,
    )


def auto_check_enabled(path: Path) -> Callable[[], bool]:
    """Re-reads the setting each time, so toggling it in Settings takes effect immediately."""

    def enabled() -> bool:
        try:
            return load_settings(path).updates.auto_check
        except SettingsError:
            return True

    return enabled


class Lazy(Generic[T]):
    """Builds its value on first use and keeps it (idle cost stays near zero until needed)."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._value: T | None = None

    def __call__(self) -> T:
        if self._value is None:
            self._value = self._factory()
        return self._value


def make_indexing_control(settings: Settings, state: StateDb) -> IndexingControl:
    """Manual start/pause of indexing; the watcher module is imported only when it is needed."""
    from localdoc_finder.watcher import SubprocessLauncher  # pulls in watchdog

    return IndexingControl(
        state,
        settings.storage.data_dir,
        SubprocessLauncher(runtime.log_dir(settings)),
        on_ac_power,
        settings.power.require_ac_power,
    )


def add_indexing_actions(
    menu: QMenu, get_control: Callable[[], IndexingControl], notify: Callable[[str], None]
) -> None:
    """A status line and one Start/Pause entry in the tray menu, refreshed when it opens."""
    status_line = menu.addAction("Indexing")
    status_line.setEnabled(False)
    toggle = menu.addAction("Start indexing")

    def refresh() -> None:
        status = get_control().status()
        status_line.setText(status.tray_summary)
        toggle.setText(
            "Pause indexing" if status.running and not status.paused else "Start indexing"
        )

    def toggled() -> None:
        control = get_control()
        status = control.status()
        notify(control.pause() if status.running and not status.paused else control.start().message)

    menu.aboutToShow.connect(refresh)
    toggle.triggered.connect(toggled)


def build_settings_window(
    settings: Settings,
    state: StateDb,
    context: Callable[[], SkillContext],
    on_hotkey: Callable[[str], None],
    updater: Updater | None = None,
    *,
    on_changed: Callable[[], None] = lambda: None,
    indexing: IndexingControl | None = None,
    forget_consent: Callable[[], None] = lambda: None,
) -> SettingsWindow:
    path = settings.settings_path()
    window = SettingsWindow(
        make_settings_controller(path, state, updater, on_changed, forget_consent),
        ModelsController(context, path),
        indexing=indexing,
    )
    window.hotkey_changed.connect(on_hotkey)
    return window


def announce_update(tray: QSystemTrayIcon, restart_action: QAction, version: str) -> None:
    restart_action.setVisible(True)
    tray.showMessage(
        "LocalDoc Finder",
        f"Version {version} is ready. Choose Restart to update in the tray menu.",
        QSystemTrayIcon.MessageIcon.Information,
        8000,
    )


class OllamaStartup(QObject):
    """Starts Ollama in the background at launch (it may take seconds); warns if it cannot.

    The check runs on a thread; the warning is a signal to this object's own slot, so the tray
    is only touched from the GUI thread.
    """

    unavailable = Signal(object)  # OllamaState: MISSING or INSTALLED_NOT_RUNNING

    def __init__(
        self,
        host: str,
        tray: QSystemTrayIcon,
        ensure: Callable[[str], OllamaState] = ensure_ollama_running,
    ) -> None:
        super().__init__(tray)  # owned by the tray, so it lives as long as the app
        self._host = host
        self._tray = tray
        self._ensure = ensure
        self.unavailable.connect(self._announce)

    def start(self) -> None:
        threading.Thread(target=self._run, name="ollama-startup", daemon=True).start()

    def _run(self) -> None:
        state = self._ensure(self._host)
        if state is not OllamaState.RUNNING:
            self.unavailable.emit(state)

    def _announce(self, state: OllamaState) -> None:
        reason = (
            "Ollama is not installed. Choose Run setup again in the tray menu to install it."
            if state is OllamaState.MISSING
            else "Ollama is installed but could not be started. Start it, then search again."
        )
        self._tray.showMessage("LocalDoc Finder", reason, QSystemTrayIcon.MessageIcon.Warning, 8000)


def setup_needed(state: StateDb) -> bool:
    """Setup has not finished, or the current Terms and Conditions have not been accepted."""
    return state.get_meta(SETUP_COMPLETED_KEY) is None or not terms_accepted(state)


def run_setup_wizard(
    settings: Settings,
    state: StateDb,
    build: FlowBuilder = build_flow,
    updater: Updater | None = None,
    on_changed: Callable[[], None] = lambda: None,
) -> None:
    """Show the first-run wizard (also reachable from the tray); returns when it closes.

    Only one wizard exists at a time: choosing "Run setup again" while one is open brings it to
    the front instead of starting a second download behind it.
    """
    if _open_wizards:
        _open_wizards[0].raise_()
        _open_wizards[0].activateWindow()
        return
    path = settings.settings_path()
    wizard = SetupWizard(
        SetupController(build, settings, state),
        make_settings_controller(path, state, updater, on_changed),
        load_catalog(settings.storage.data_dir),
        probe_hardware(),
        record_terms=lambda: accept_terms(state),
    )
    _open_wizards.append(wizard)
    try:
        wizard.exec()
    finally:
        _open_wizards.remove(wizard)


def hotkey_applier(hotkey: HotkeyFilter, tray: QSystemTrayIcon) -> Callable[[str], None]:
    """Re-register the global hotkey right away when the user changes it in Settings."""

    def apply(spec: str) -> None:
        try:
            ok = hotkey.register(spec)  # the old key stays active unless the new one is secured
        except ValueError:
            ok = False
        if ok:
            tray.setToolTip(f"LocalDoc Finder ({spec})")
        else:
            current = hotkey.spec or "no hotkey"
            tray.setToolTip(f"LocalDoc Finder ({current}) - {spec} is unavailable")

    return apply


def show_settings_error(message: str) -> None:
    QMessageBox.critical(None, "LocalDoc Finder", message)


def load_settings_or_report(show: Callable[[str], None] | None = None) -> Settings | None:
    """The settings, or ``None`` after telling the user why the app cannot start."""
    try:
        return load_settings()
    except SettingsError as exc:
        logger.error("settings are invalid: %s", exc)
        (show or show_settings_error)(
            f"LocalDoc Finder cannot start because its settings are invalid.\n\n{exc}"
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
    use_app_icon(app)
    settings = load_settings_or_report()
    if settings is None:
        return EXIT_BAD_SETTINGS
    configure_logging("app", runtime.log_dir(settings), settings.log_level)
    apply_theme(app, settings.app.theme)
    with single_instance("app", settings.storage.data_dir) as acquired:
        if not acquired:
            logger.info("another copy of the app is already running")
            return EXIT_OK
        return run_app(app, settings, args)


def register_hotkey(hotkey: HotkeyFilter, spec: str) -> bool:
    try:
        return hotkey.register(spec)
    except ValueError:
        logger.exception("invalid hotkey %r", spec)
        return False


def announce(tray: QSystemTrayIcon, text: str) -> None:
    tray.showMessage("LocalDoc Finder", text, QSystemTrayIcon.MessageIcon.Information, 5000)


def hotkey_tooltip(spec: str, registered: bool) -> str:
    return f"LocalDoc Finder ({spec})" + ("" if registered else " - hotkey unavailable")


def leave_without_terms(hotkey: HotkeyFilter, tray: QSystemTrayIcon) -> None:
    logger.info("the terms were not accepted; exiting")
    hotkey.unregister()
    tray.hide()


def warn_hotkey_unavailable(tray: QSystemTrayIcon, spec: str) -> None:
    tray.showMessage(
        "LocalDoc Finder",
        f"Could not register {spec}; use the tray icon.",
        QSystemTrayIcon.MessageIcon.Warning,
        5000,
    )


class SettingsOpener:
    """Shows the Settings window, building it on first use (idle cost stays near zero)."""

    def __init__(self, build: Callable[[], SettingsWindow]) -> None:
        self._build = build
        self._window: SettingsWindow | None = None

    def _get(self) -> SettingsWindow:
        if self._window is None:
            self._window = self._build()
        return self._window

    def open(self, _checked: bool = False) -> None:  # a menu action passes ``checked``
        self._get().open()

    def open_cloud(self) -> None:
        window = self._get()
        window.open(window.cloud)


def settings_applier(
    app: QApplication, context: ContextFactory, window: SearchWindow, settings_path: Path
) -> Callable[[], None]:
    """What to do after a setting is saved, so no restart is needed."""

    def apply() -> None:
        context.invalidate()
        window.reload_context()
        try:
            window.apply_scheme(apply_theme(app, load_settings(settings_path).app.theme))
        except SettingsError:
            logger.exception("could not re-read the theme")

    return apply


def run_app(app: QApplication, settings: Settings, args: argparse.Namespace) -> int:
    with StateDb(settings.storage.data_dir) as state:
        settings_path = settings.settings_path()
        context = make_context_factory(settings, state, lambda: load_settings(settings_path))
        updater = Updater(resolve_source(settings.updates.repo_url), state=state)
        window = build_window(settings, state, context)

        settings_changed = settings_applier(app, context, window, settings_path)
        window.cloud_switcher = CloudSwitcher(
            make_settings_controller(
                settings_path, state, None, settings_changed, context.consent.revoke_session
            )
        )

        hotkey = HotkeyFilter(window.summon)
        app.installNativeEventFilter(hotkey)
        registered = register_hotkey(hotkey, settings.search.hotkey)

        tray = QSystemTrayIcon(tray_icon(), app)
        menu = QMenu()
        indexing_control = Lazy(lambda: make_indexing_control(settings, state))
        settings_opener = SettingsOpener(
            lambda: build_settings_window(
                settings,
                state,
                context,
                hotkey_applier(hotkey, tray),
                updater,
                on_changed=settings_changed,
                indexing=indexing_control(),
                forget_consent=context.consent.revoke_session,
            )
        )
        window.settings_requested.connect(settings_opener.open)  # the gear in the popup's header
        window.cloud_settings_requested.connect(settings_opener.open_cloud)
        menu.addAction("Search", window.summon)
        menu.addAction("Settings…", settings_opener.open)
        add_indexing_actions(menu, indexing_control, lambda text: announce(tray, text))
        menu.addAction(
            "Run setup again…",
            lambda: run_setup_wizard(settings, state, updater=updater, on_changed=settings_changed),
        )
        restart_action = menu.addAction("Restart to update", updater.restart_to_update)
        restart_action.setVisible(False)  # shown once an update has been downloaded
        menu.addAction("Quit", app.quit)
        app.aboutToQuit.connect(window.shutdown)  # every way of quitting unloads the models
        tray.setContextMenu(menu)
        tray.setToolTip(hotkey_tooltip(settings.search.hotkey, registered))
        tray.activated.connect(
            lambda reason: (
                window.summon() if reason == QSystemTrayIcon.ActivationReason.Trigger else None
            )
        )
        tray.show()
        if not registered:
            warn_hotkey_unavailable(tray, settings.search.hotkey)
        if args.setup or setup_needed(state):
            announce(tray, "Setup is open. It downloads the search models, so keep it running.")
            run_setup_wizard(settings, state, updater=updater, on_changed=settings_changed)
        if not terms_accepted(state):  # declined or closed: nothing may index or update
            leave_without_terms(hotkey, tray)
            return EXIT_OK
        OllamaStartup(settings.ollama_host, tray).start()
        scheduler = UpdateScheduler(updater, auto_check_enabled(settings.settings_path()))
        scheduler.ready.connect(lambda version: announce_update(tray, restart_action, version))
        scheduler.start()
        if is_frozen():
            start_watcher()  # a fresh install has had no logon yet; a no-op if one is running
        if args.show:
            window.summon()
        code = app.exec()  # ``aboutToQuit`` fires inside this call, before it returns
        hotkey.unregister()
        with suppress(RuntimeError, TypeError):  # the window is going away; drop the connection
            app.aboutToQuit.disconnect(window.shutdown)
    return code


if __name__ == "__main__":
    sys.exit(main())
