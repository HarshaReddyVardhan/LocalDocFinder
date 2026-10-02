import os
import sys
import time

import pytest

import indexer_config as cfg
import power
import watcher


def wait_for(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def w(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    wt = watcher.Watcher(tmp_path / "data", [str(root)],
                         worker_cmd=[sys.executable, "-c", "import time; time.sleep(0.2)"])
    wt.start_observers()
    yield wt, root
    wt.observer.stop()
    wt.observer.join(timeout=5)
    wt.state.close()


def queued(state):
    return {os.path.basename(r[0]): r[1] for r in state.sql.execute("SELECT path, op FROM queue")}


def test_live_events_are_queued_with_debounce(w):
    wt, root = w
    f = root / "a.py"
    f.write_text("x = 1\n")
    assert wait_for(lambda: "a.py" in queued(wt.state))
    assert wt.state.claim(10) == []                        # debounced: not due yet
    assert len(wt.state.claim(10, ignore_debounce=True)) == 1


def test_noise_is_not_queued(w):
    wt, root = w
    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("a")
    (root / "ok.py").write_text("x = 1\n")
    (root / "package-lock.json").write_text("{}")
    (root / ".env").write_text("SECRET=1")
    assert wait_for(lambda: "ok.py" in queued(wt.state))
    time.sleep(0.5)
    assert set(queued(wt.state)) == {"ok.py"}


def test_delete_event_queues_delete_only_for_indexed_files(w):
    wt, root = w
    f = root / "gone.py"
    f.write_text("x = 1\n")
    assert wait_for(lambda: "gone.py" in queued(wt.state))
    wt.state.sql.execute("DELETE FROM queue")
    wt.state.manifest_set(str(f), 1, 1, "h")  # pretend it was indexed
    f.unlink()
    assert wait_for(lambda: queued(wt.state).get("gone.py") == "delete")


def test_directory_delete_queues_children(w):
    wt, root = w
    d = root / "pkg"
    d.mkdir()
    for n in ("a.py", "b.py"):
        wt.state.manifest_set(str(d / n), 1, 1, "h")
    wt._handler = watcher.ChangeHandler(wt.state, wt.projects)
    from watchdog.events import DirDeletedEvent
    wt._handler.on_deleted(DirDeletedEvent(str(d)))
    assert queued(wt.state) == {"a.py": "delete", "b.py": "delete"}


def test_battery_means_queue_only_no_worker(w, monkeypatch, caplog):
    wt, root = w
    wt.state.enqueue(str(root / "a.py"), delay=-1)  # due
    monkeypatch.setattr(power, "on_ac_power", lambda: False)
    monkeypatch.setattr(power, "seconds_since_input", lambda: 1e9)
    for _ in range(3):
        wt.tick()
    assert wt.proc is None
    assert wt.state.queue_size() == 1                   # still queued, nothing processed
    assert "battery, queue only" in wt.gate.ready()[1]


def test_ac_needs_settle_time_then_spawns(w, monkeypatch):
    wt, root = w
    wt.state.enqueue(str(root / "a.py"), delay=-1)
    wt.state.set_meta("last_reconcile", str(time.time()))
    monkeypatch.setattr(power, "on_ac_power", lambda: True)
    monkeypatch.setattr(power, "seconds_since_input", lambda: 1e9)
    monkeypatch.setattr(power, "fullscreen_app_active", lambda: False)
    monkeypatch.setattr(power, "gpu_utilization", lambda: 0)
    monkeypatch.setattr(cfg, "IDLE_CPU_PERCENT", 101)
    monkeypatch.setattr(cfg, "IDLE_CPU_SECONDS", 0)
    monkeypatch.setattr(cfg, "AC_SETTLE_SECONDS", 3600)
    wt.tick()
    assert wt.proc is None and "settle" in wt.gate.ready()[1]
    monkeypatch.setattr(cfg, "AC_SETTLE_SECONDS", 0)
    wt.tick()
    assert wt.proc is not None
    wt.proc.wait(timeout=10)


def test_unplug_terminates_stuck_worker(w, monkeypatch):
    wt, root = w
    import subprocess
    wt.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(power, "on_ac_power", lambda: False)
    monkeypatch.setattr(watcher, "unload_model", lambda m: None)
    wt.tick()
    wt.unplugged_at -= 30  # 30 s of grace elapsed
    wt.tick()
    assert wt.proc.wait(timeout=10) is not None


def test_status_prints(tmp_path, capsys):
    watcher.status(tmp_path)
    assert "on AC power" in capsys.readouterr().out
