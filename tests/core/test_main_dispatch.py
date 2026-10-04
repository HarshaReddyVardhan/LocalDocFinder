import pytest

from localdoc_finder import __main__ as entry
from localdoc_finder import watcher, worker
from localdoc_finder.app import main as app_main


@pytest.mark.parametrize(
    ("argv", "expected_main", "expected_args"),
    [
        (["watcher", "--status"], watcher.main, ["--status"]),
        (["worker", "--reconcile"], worker.main, ["--reconcile"]),
        (["app", "--show"], app_main.main, ["--show"]),
        (["setup"], app_main.main, ["--setup"]),
        ([], app_main.main, []),
        (["--show"], app_main.main, ["--show"]),  # no entry named: the app, with its own flags
    ],
)
def test_dispatch_picks_the_entry_point(
    argv: list[str], expected_main: object, expected_args: list[str]
) -> None:
    handler, args = entry.resolve(argv)
    assert handler is expected_main
    assert args == expected_args


def test_main_runs_the_resolved_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(worker, "main", lambda args: seen.append(list(args)) or 5)
    assert entry.main(["worker", "--now"]) == 5
    assert seen == [["--now"]]


def test_main_reads_sys_argv_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["LocalDocFinder.exe", "watcher", "--status"])
    monkeypatch.setattr(watcher, "main", lambda args: 0 if list(args) == ["--status"] else 9)
    assert entry.main() == 0


def test_frozen_build_runs_the_velopack_hooks_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from localdoc_finder.core import lifecycle

    order: list[str] = []
    monkeypatch.setattr(entry, "is_frozen", lambda: True)
    monkeypatch.setattr(lifecycle, "run_startup_hooks", lambda **_kw: order.append("hooks"))
    monkeypatch.setattr(worker, "main", lambda args: order.append("worker") or 0)
    assert entry.main(["worker"]) == 0
    assert order == ["hooks", "worker"]


def test_unfrozen_runs_skip_the_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    from localdoc_finder.core import lifecycle

    monkeypatch.setattr(entry, "is_frozen", lambda: False)
    monkeypatch.setattr(
        lifecycle, "run_startup_hooks", lambda **_kw: pytest.fail("hooks must not run from source")
    )
    monkeypatch.setattr(worker, "main", lambda args: 0)
    assert entry.main(["worker"]) == 0


def test_autostart_follows_the_users_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from localdoc_finder.core import lifecycle
    from localdoc_finder.core.settings import Settings, SettingsError

    seen: list[bool] = []
    monkeypatch.setattr(lifecycle, "run_startup_hooks", lambda *, enabled: seen.append(enabled()))
    monkeypatch.setattr(
        "localdoc_finder.core.settings.load_settings",
        lambda: Settings(app={"start_with_windows": False}),  # type: ignore[arg-type]
    )
    entry._run_velopack_hooks()

    def broken() -> Settings:
        raise SettingsError("bad file")

    monkeypatch.setattr("localdoc_finder.core.settings.load_settings", broken)
    entry._run_velopack_hooks()
    assert seen == [False, True]  # an unreadable settings file must not block registration
