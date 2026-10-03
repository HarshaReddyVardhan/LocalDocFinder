import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon
from tests.core.app.test_models_panel import wait_for

from vector_embed.app import main as app_main
from vector_embed.app.update_scheduler import UpdateScheduler
from vector_embed.core.store.sqlite import StateDb
from vector_embed.core.updates import Updater


class FakeManager:
    def __init__(self, latest: str | None) -> None:
        self.latest = latest
        self.checks = 0

    def get_current_version(self) -> str:
        return "1.0.0"

    def check_for_updates(self) -> object | None:
        self.checks += 1
        if not self.latest:
            return None
        return SimpleNamespace(TargetFullRelease=SimpleNamespace(Version=self.latest))

    def download_updates(self, info: object, progress: object = None) -> None:
        return None

    def apply_updates_and_restart(self, update: object) -> None:
        return None


def scheduler_for(
    tmp_path: Path, manager: FakeManager, enabled: bool = True
) -> tuple[UpdateScheduler, Updater]:
    updater = Updater(
        "https://github.com/x/y", state=StateDb(tmp_path), factory=lambda _url: manager
    )
    return UpdateScheduler(updater, lambda: enabled, startup_delay_ms=0), updater


def test_a_ready_update_is_announced_once(qapp: QApplication, tmp_path: Path) -> None:
    manager = FakeManager("1.1.0")
    scheduler, updater = scheduler_for(tmp_path, manager)
    versions: list[str] = []
    scheduler.ready.connect(versions.append)
    scheduler.tick()
    wait_for(qapp, lambda: bool(versions))
    assert versions == ["1.1.0"]
    scheduler._on_done(updater.outcome)  # the same version again: no second notification
    assert versions == ["1.1.0"]


def test_nothing_runs_when_auto_check_is_off(qapp: QApplication, tmp_path: Path) -> None:
    manager = FakeManager("1.1.0")
    scheduler, _ = scheduler_for(tmp_path, manager, enabled=False)
    scheduler.tick()
    qapp.processEvents()
    assert manager.checks == 0


def test_a_check_is_only_made_once_a_day(qapp: QApplication, tmp_path: Path) -> None:
    manager = FakeManager(None)
    scheduler, _ = scheduler_for(tmp_path, manager)
    scheduler.tick()
    wait_for(qapp, lambda: manager.checks == 1 and not scheduler._busy)
    scheduler.tick()
    qapp.processEvents()
    assert manager.checks == 1


def test_no_second_check_while_one_is_running(qapp: QApplication, tmp_path: Path) -> None:
    manager = FakeManager(None)
    scheduler, _ = scheduler_for(tmp_path, manager)
    scheduler._busy = True
    scheduler.tick()
    qapp.processEvents()
    assert manager.checks == 0


def test_start_arms_the_hourly_timer(qapp: QApplication, tmp_path: Path) -> None:
    scheduler, _ = scheduler_for(tmp_path, FakeManager(None))
    scheduler.start()
    assert scheduler._timer.isActive()
    scheduler.stop()
    assert not scheduler._timer.isActive()


def test_announcing_shows_the_restart_action_and_a_message(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(
        QSystemTrayIcon, "showMessage", lambda _self, _t, text, *_a: messages.append(text)
    )
    tray = QSystemTrayIcon()
    menu = QMenu()
    action = menu.addAction("Restart to update")
    action.setVisible(False)
    app_main.announce_update(tray, action, "1.1.0")
    assert action.isVisible()
    assert "1.1.0" in messages[0]


def test_auto_check_setting_is_read_fresh(tmp_path: Path) -> None:
    from vector_embed.core.settings_io import set_setting

    path = tmp_path / "settings.toml"
    enabled = app_main.auto_check_enabled(path)
    assert enabled()  # no file: the default is on
    set_setting(path, ["updates", "auto_check"], False)
    assert not enabled()
    path.write_text("not = [valid", encoding="utf-8")
    assert enabled()  # an unreadable file must not silently turn updates off


def test_a_crashing_check_still_clears_the_busy_flag(qapp: QApplication) -> None:
    class Exploding:
        def due(self) -> bool:
            return True

        def check(self) -> object:
            raise MemoryError("native layer blew up")

    scheduler = UpdateScheduler(Exploding(), lambda: True, startup_delay_ms=10**9)  # type: ignore[arg-type]
    scheduler.tick()
    deadline = time.time() + 5
    while scheduler._busy and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not scheduler._busy  # without this, no update check would ever run again
    scheduler.tick()
    assert scheduler._busy  # and the next hourly tick can start one
