import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from localdoc_finder.core import updates
from localdoc_finder.core.store.sqlite import StateDb
from localdoc_finder.core.updates import (
    CHECK_INTERVAL_SECONDS,
    LAST_CHECK_KEY,
    UpdateKind,
    Updater,
)

REPO = "https://github.com/example/app"


def release(version: str) -> SimpleNamespace:
    return SimpleNamespace(TargetFullRelease=SimpleNamespace(Version=version))


class FakeManager:
    def __init__(self, latest: str | None = None) -> None:
        self.latest = latest
        self.downloaded: list[object] = []
        self.applied: list[object] = []
        self.fail: Exception | None = None
        self.checks = 0

    def get_current_version(self) -> str:
        return "1.0.0"

    def check_for_updates(self) -> object | None:
        self.checks += 1
        if self.fail:
            raise self.fail
        return release(self.latest) if self.latest else None

    def download_updates(self, info: object, progress: object = None) -> None:
        self.downloaded.append(info)

    def apply_updates_and_restart(self, update: object) -> None:
        self.applied.append(update)


@pytest.fixture
def state(tmp_path: Path) -> StateDb:
    return StateDb(tmp_path)


def make(manager: FakeManager, state: StateDb | None = None, now: float = 1000.0) -> Updater:
    return Updater(REPO, state=state, factory=lambda _url: manager, clock=lambda: now)


def test_up_to_date() -> None:
    outcome = make(FakeManager(None)).check()
    assert outcome.kind is UpdateKind.UP_TO_DATE
    assert "1.0.0" in outcome.message


def test_new_release_is_downloaded_and_ready() -> None:
    manager = FakeManager("1.1.0")
    updater = make(manager)
    outcome = updater.check()
    assert outcome.kind is UpdateKind.READY
    assert outcome.version == "1.1.0"
    assert "Restart to update" in outcome.message
    assert len(manager.downloaded) == 1
    assert updater.outcome == outcome


def test_a_downloaded_update_is_not_fetched_twice() -> None:
    manager = FakeManager("1.1.0")
    updater = make(manager)
    updater.check()
    again = updater.check()
    assert again.kind is UpdateKind.READY
    assert manager.checks == 1
    assert len(manager.downloaded) == 1


def test_restart_applies_the_pending_update() -> None:
    manager = FakeManager("1.1.0")
    updater = make(manager)
    updater.check()
    updater.restart_to_update()
    assert len(manager.applied) == 1


def test_restart_without_a_download_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="no update"):
        make(FakeManager("1.1.0")).restart_to_update()


def test_no_source_means_not_configured() -> None:
    outcome = Updater("", factory=lambda _url: FakeManager()).check()
    assert outcome.kind is UpdateKind.NOT_CONFIGURED


def test_a_dev_checkout_is_not_installed() -> None:
    def factory(_url: str) -> FakeManager:
        raise RuntimeError("This application is not properly installed")

    outcome = Updater(REPO, factory=factory).check()
    assert outcome.kind is UpdateKind.NOT_INSTALLED
    assert "installed app" in outcome.message


@pytest.mark.parametrize("error", [RuntimeError("net down"), OSError("dns"), ValueError("bad")])
def test_network_problems_become_a_failed_outcome(error: Exception) -> None:
    manager = FakeManager("1.1.0")
    manager.fail = error
    outcome = make(manager).check()
    assert outcome.kind is UpdateKind.FAILED
    assert str(error) in outcome.message


def test_a_failed_check_is_retried_not_postponed_a_day(state: StateDb) -> None:
    manager = FakeManager("1.1.0")
    manager.fail = RuntimeError("net down")
    updater = make(manager, state)
    updater.check()
    assert state.get_meta(LAST_CHECK_KEY) is None
    assert updater.due()


def test_due_follows_the_last_check(state: StateDb) -> None:
    assert make(FakeManager(), state, now=1000.0).due()  # never checked
    make(FakeManager(), state, now=1000.0).check()
    assert state.get_meta(LAST_CHECK_KEY) == "1000.0"
    assert not make(FakeManager(), state, now=1000.0 + CHECK_INTERVAL_SECONDS - 1).due()
    assert make(FakeManager(), state, now=1000.0 + CHECK_INTERVAL_SECONDS).due()


def test_due_without_state_is_always_true() -> None:
    assert make(FakeManager()).due()


def test_source_prefers_the_users_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updates, "bundled_source", lambda: "https://github.com/built/in")
    assert (
        updates.resolve_source(" https://github.com/mine/repo ") == "https://github.com/mine/repo"
    )
    assert updates.resolve_source("") == "https://github.com/built/in"


def test_bundled_source_is_empty_in_a_dev_checkout() -> None:
    assert updates.bundled_source() == ""


def test_velopack_manager_builds_a_github_source(monkeypatch: pytest.MonkeyPatch) -> None:
    import velopack

    seen: list[object] = []
    monkeypatch.setattr(velopack, "GithubSource", lambda url: seen.append(url) or "source")
    monkeypatch.setattr(velopack, "UpdateManager", lambda source: ("manager", source))
    assert updates.velopack_manager(REPO) == ("manager", "source")
    assert seen == [REPO]


class TestOneCheckAtATime:
    def test_overlapping_checks_do_not_run_together(self) -> None:
        import threading

        release_first = threading.Event()
        inside: list[int] = []
        peak = [0]
        running = [0]

        class Slow(FakeManager):
            def check_for_updates(self) -> object | None:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
                inside.append(1)
                release_first.wait(5)
                running[0] -= 1
                return None

        updater = Updater(REPO, factory=lambda _url: Slow())
        results: list[object] = []
        threads = [
            threading.Thread(target=lambda: results.append(updater.check())) for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        deadline = time.time() + 2
        while not inside and time.time() < deadline:
            time.sleep(0.01)
        release_first.set()
        for thread in threads:
            thread.join(timeout=5)
        assert len(results) == 2
        assert peak[0] == 1  # never two native checks at once

    def test_an_unexpected_native_error_is_a_failed_outcome_not_an_exception(self) -> None:
        manager = FakeManager("2.0.0")
        manager.fail = KeyError("something nobody listed")  # not RuntimeError, OSError, ValueError
        outcome = Updater(REPO, factory=lambda _url: manager).check()
        assert outcome.kind is UpdateKind.FAILED
        assert "something nobody listed" in outcome.message
        # and the lock was released: the next check works
        manager.fail = None
        assert Updater(REPO, factory=lambda _url: manager).check().kind is UpdateKind.READY


def test_ready_outcome_carries_the_release_notes() -> None:
    manager = FakeManager("1.1.0")

    def with_notes() -> object:
        return SimpleNamespace(
            TargetFullRelease=SimpleNamespace(Version="1.1.0", NotesMarkdown="  ## New\n- Thing\n")
        )

    manager.check_for_updates = with_notes  # type: ignore[method-assign]  # fake returns notes
    outcome = make(manager).check()
    assert outcome.notes == "## New\n- Thing"


def test_a_release_without_notes_has_empty_notes() -> None:
    assert make(FakeManager("1.1.0")).check().notes == ""
