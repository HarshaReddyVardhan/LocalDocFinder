import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QWizard
from tests.core.app.test_models_panel import wait_for
from tests.core.app.test_settings_window import FakeKeys
from tests.core.conftest import Env
from tests.core.setup.fakes import GPU8, Harness

from vector_embed.app import main as app_main
from vector_embed.app.settings_controller import SettingsController
from vector_embed.app.setup_controller import SetupController
from vector_embed.app.setup_wizard import ModelsPage, SetupWizard
from vector_embed.core.models.benchmark import BenchKind
from vector_embed.core.models.catalog import load_catalog
from vector_embed.core.models.hardware import Hardware
from vector_embed.core.settings import Settings
from vector_embed.core.setup.flow import (
    SETUP_COMPLETED_KEY,
    SetupEvent,
    SetupFlow,
    SetupOptions,
    SetupResult,
    SlowOffer,
    Stage,
)
from vector_embed.core.setup.plan import SetupChoices

GPU4 = Hardware("GTX 1650", 4096, 3500, 16000, 8000, 8, True)
NO_GPU = Hardware(None, 0, 0, 16000, 8000, 8, True)


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def builder(harness: Harness, hardware: Hardware = GPU8) -> Callable[..., SetupFlow]:
    def build(
        settings: Settings,
        state: object,
        progress: Callable[[SetupEvent], None],
        accept: Callable[[SlowOffer], bool],
    ) -> SetupFlow:
        return harness.flow(hardware, progress=progress, accept=accept)

    return build


def controller_for(harness: Harness, hardware: Hardware = GPU8) -> SetupController:
    return SetupController(builder(harness, hardware), Settings(), harness.state)  # type: ignore[arg-type]


def wizard_for(
    harness: Harness, hardware: Hardware = GPU8, ask: Callable[[SlowOffer], bool] | None = None
) -> SetupWizard:
    settings_controller = SettingsController(harness.settings_path, harness.state, FakeKeys())
    return SetupWizard(
        controller_for(harness, hardware),
        settings_controller,
        load_catalog(),
        hardware,
        ask_downgrade=ask or (lambda offer: True),
    )


# ------------------------------------------------------------------ controller
def test_controller_runs_the_flow_and_reports(qapp: QApplication, harness: Harness) -> None:
    controller = controller_for(harness)
    events: list[SetupEvent] = []
    results: list[SetupResult] = []
    controller.progressed.connect(events.append)
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(results))
    assert results[0].chat_model == "qwen3.5:9b"
    assert Stage.PULL in {e.stage for e in events}
    assert not controller.running
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is not None


def test_controller_ignores_a_second_start_while_running(
    qapp: QApplication, harness: Harness
) -> None:
    controller = controller_for(harness)
    results: list[SetupResult] = []
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(results))
    qapp.processEvents()
    assert len(results) == 1


def test_controller_reports_failures_and_can_retry(qapp: QApplication, harness: Harness) -> None:
    harness.host.fail_on = "qwen3.5:9b"
    controller = controller_for(harness)
    failures: list[str] = []
    results: list[SetupResult] = []
    controller.failed.connect(failures.append)
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(failures))
    assert "network down" in failures[0]
    harness.host.fail_on = None
    wait_for(qapp, lambda: not controller.running)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(results))
    assert results[0].chat_model == "qwen3.5:9b"


def test_controller_surfaces_unexpected_errors(qapp: QApplication, harness: Harness) -> None:
    controller = controller_for(harness)
    failures: list[str] = []
    controller.failed.connect(failures.append)

    def explode(model: str) -> None:
        raise ZeroDivisionError("bug")

    harness.host.pull = explode  # type: ignore[method-assign,assignment]
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(failures))
    assert failures == ["ZeroDivisionError: bug"]


def test_downgrade_question_round_trips_through_the_gui_thread(
    qapp: QApplication, harness: Harness
) -> None:
    harness.model_rates = {"qwen3.5:9b": 2.0, "qwen3:8b": 25.0}
    controller = controller_for(harness)
    offers: list[SlowOffer] = []

    def answer(offer: SlowOffer) -> None:
        offers.append(offer)
        controller.answer_downgrade(True)

    controller.downgrade_offered.connect(answer)
    results: list[SetupResult] = []
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(results))
    assert [o.alternative for o in offers] == ["qwen3:8b"]
    assert results[0].chat_model == "qwen3:8b"


# ------------------------------------------------------------------ pages
def load_models_page(qapp: QApplication, page: ModelsPage) -> None:
    """Show the page and wait for the background look at Ollama and the disk to finish."""
    page.initializePage()
    wait_for(qapp, lambda: page.embed.count() > 0 and "Checking" not in page.disk.text())


def test_welcome_describes_the_hardware(qapp: QApplication, harness: Harness) -> None:
    gpu_wizard = wizard_for(harness)
    cpu_wizard = wizard_for(harness, NO_GPU)
    assert "RTX 2070" in gpu_wizard.welcome.summary.text()
    assert "CPU" in cpu_wizard.welcome.summary.text()


def test_ollama_page_requires_consent_to_install(qapp: QApplication, tmp_path: Path) -> None:
    harness = Harness(tmp_path, up=False)
    wizard = wizard_for(harness)
    page = wizard.ollama
    page.initializePage()
    wait_for(qapp, lambda: "not installed" in page.status.text())
    assert "not installed" in page.status.text()
    assert not page.isComplete()
    assert not page.install_consented
    page.consent.setChecked(True)
    assert page.isComplete()
    assert page.install_consented


def test_ollama_page_needs_nothing_when_it_is_running(qapp: QApplication, harness: Harness) -> None:
    wizard = wizard_for(harness)
    page = wizard.ollama
    page.initializePage()
    wait_for(qapp, lambda: "running" in page.status.text())
    assert page.isComplete()
    assert not page.install_consented
    assert "running" in page.status.text()


def test_ollama_page_starts_an_installed_but_stopped_server(
    qapp: QApplication, tmp_path: Path
) -> None:
    harness = Harness(tmp_path, up=False)
    harness.system.exe = Path("ollama.exe")
    wizard = wizard_for(harness)
    page = wizard.ollama
    page.initializePage()
    wait_for(qapp, lambda: "will start it" in page.status.text())
    assert "will start it" in page.status.text()
    assert page.isComplete()


def test_models_page_is_prefilled_with_the_auto_picks(qapp: QApplication, harness: Harness) -> None:
    wizard = wizard_for(harness)
    page = wizard.models
    load_models_page(qapp, page)
    assert page.embed.currentData() == "qwen3-embedding:0.6b"
    assert page.chat.currentData() == "qwen3.5:9b"
    assert page.chat.currentText() == "qwen3.5:9b (6100 MB)"
    assert page.choices() == SetupChoices("qwen3-embedding:0.6b", "qwen3.5:9b", ())
    assert "To download: 6740 MB" in page.disk.text()
    assert page.isComplete()


def test_models_page_lists_every_catalog_model_and_flags_misfits(
    qapp: QApplication, harness: Harness
) -> None:
    wizard = wizard_for(harness, GPU4)
    page = wizard.models
    load_models_page(qapp, page)
    names = [page.chat.itemText(i) for i in range(page.chat.count())]
    assert len(names) == len(load_catalog().preferences("chat"))
    assert "qwen3.5:9b (6100 MB) - won't fit" in names
    assert page.chat.currentData() == "llama3.2"


def test_models_page_updates_the_download_size_for_choices(
    qapp: QApplication, harness: Harness
) -> None:
    wizard = wizard_for(harness)
    page = wizard.models
    load_models_page(qapp, page)
    page.chat.setCurrentIndex(page.chat.findData("llama3.2"))
    page.extras["caption"].setChecked(True)
    assert page.choices().chat == "llama3.2"
    assert page.choices().extras == ("caption",)
    assert "To download: 5840 MB" in page.disk.text()  # 640 + 2000 + 3200


def test_models_page_blocks_when_the_disk_is_too_small(
    qapp: QApplication, harness: Harness
) -> None:
    harness.system.free_mb = 3000
    wizard = wizard_for(harness)
    page = wizard.models
    load_models_page(qapp, page)
    assert not page.isComplete()
    assert "Not enough free space" in page.disk.text()


# ------------------------------------------------------------------ whole wizard
def test_wizard_runs_the_whole_setup(qapp: QApplication, harness: Harness) -> None:
    wizard = wizard_for(harness)
    wizard.restart()
    assert wizard.currentPage() is wizard.welcome
    wizard.next()  # -> Ollama
    wizard.next()  # -> Models
    assert wizard.currentPage() is wizard.models
    wizard.next()  # commit: starts the download
    wait_for(qapp, wizard.speed.isComplete)
    assert wizard.currentPage() is wizard.speed
    assert "qwen3.5:9b: 30.0 tok/s" in wizard.speed.results.text()
    assert harness.host.pulled == ["qwen3-embedding:0.6b", "qwen3.5:9b"]
    wizard.next()  # -> settings
    assert wizard.currentPage() is wizard.settings_page
    assert wizard.settings_page.general.hotkey.text() == "ctrl+alt+space"
    wizard.next()  # -> done
    assert "ctrl+alt+space" in wizard.done_page.text.text()
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is not None


def test_wizard_asks_before_swapping_a_slow_model(qapp: QApplication, harness: Harness) -> None:
    harness.model_rates = {"qwen3.5:9b": 2.0, "qwen3:8b": 25.0}
    asked: list[SlowOffer] = []

    def ask(offer: SlowOffer) -> bool:
        asked.append(offer)
        return True

    wizard = wizard_for(harness, ask=ask)
    wizard.restart()
    wizard.next()
    wizard.next()
    wizard.next()
    wait_for(qapp, wizard.speed.isComplete)
    assert [o.alternative for o in asked] == ["qwen3:8b"]
    assert "qwen3:8b" in wizard.speed.results.text()
    assert harness.saved()["models"] == {"overrides": {"chat": "qwen3:8b"}}


def test_wizard_shows_a_failure_and_retries(qapp: QApplication, harness: Harness) -> None:
    harness.host.fail_on = "qwen3.5:9b"
    wizard = wizard_for(harness)
    wizard.restart()
    wizard.next()
    wizard.next()
    wizard.next()
    wait_for(qapp, lambda: not wizard.download.retry.isHidden())
    assert "network down" in wizard.download.line.text()
    assert not wizard.download.isComplete()
    harness.host.fail_on = None
    wait_for(qapp, lambda: not wizard._controller.running)
    wizard.download.retry.click()
    wait_for(qapp, wizard.speed.isComplete)
    assert wizard.currentPage() is wizard.speed
    assert wizard.download.retry.isHidden()


def test_wizard_button_is_labelled_download(qapp: QApplication, harness: Harness) -> None:
    wizard = wizard_for(harness)
    assert wizard.buttonText(QWizard.WizardButton.CommitButton) == "Download"
    assert wizard.models.isCommitPage()
    assert harness.rates[BenchKind.CHAT] == 30.0


# ------------------------------------------------------------------ app wiring
def test_setup_needed_follows_the_completion_marker(harness: Harness) -> None:
    assert app_main.setup_needed(harness.state)
    harness.state.set_meta(SETUP_COMPLETED_KEY, "1.0")
    assert not app_main.setup_needed(harness.state)


def test_run_setup_wizard_builds_and_executes_the_wizard(
    qapp: QApplication, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran: list[SetupWizard] = []
    monkeypatch.setattr(SetupWizard, "exec", lambda self: ran.append(self) or 0)
    monkeypatch.setattr(app_main, "KeyringStore", FakeKeys)
    monkeypatch.setattr(app_main, "probe_hardware", lambda: GPU8)
    settings = Settings(storage={"data_dir": harness.tmp_path / "data"})  # type: ignore[arg-type]
    app_main.run_setup_wizard(settings, harness.state, builder(harness))
    assert len(ran) == 1


@pytest.mark.parametrize("needed", [True, False])
def test_main_runs_the_wizard_only_when_setup_is_incomplete(
    qapp: QApplication,
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
    needed: bool,
) -> None:
    if not needed:
        env.state.set_meta(SETUP_COMPLETED_KEY, "1.0")
    calls: list[int] = []
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: 0)
    monkeypatch.setattr(app_main.HotkeyFilter, "register", lambda _self, _spec: True)
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: None)
    monkeypatch.setattr(app_main, "run_setup_wizard", lambda *_a, **_k: calls.append(1))
    assert app_main.main([]) == 0
    assert calls == ([1] if needed else [])


def test_main_setup_flag_forces_the_wizard(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.state.set_meta(SETUP_COMPLETED_KEY, "1.0")
    calls: list[int] = []
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: 0)
    monkeypatch.setattr(app_main.HotkeyFilter, "register", lambda _self, _spec: True)
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: None)
    monkeypatch.setattr(app_main, "run_setup_wizard", lambda *_a, **_k: calls.append(1))
    assert app_main.main(["--setup"]) == 0
    assert calls == [1]


def test_settings_controller_applies_autostart_through_task_scheduler(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    applied: list[bool] = []
    monkeypatch.setattr(app_main.Autostart, "apply", lambda _self, enabled: applied.append(enabled))
    monkeypatch.setattr(app_main, "KeyringStore", FakeKeys)
    controller = app_main.make_settings_controller(env.data_dir / "settings.toml", env.state)
    controller.set_start_with_windows(False)
    assert applied == [False]


def test_a_frozen_app_starts_the_watcher_and_the_update_scheduler(
    qapp: QApplication, env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    env.state.set_meta(SETUP_COMPLETED_KEY, "1.0")
    started: list[str] = []
    monkeypatch.setattr(app_main, "load_settings", lambda: env.settings)
    monkeypatch.setattr(app_main, "configure_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app_main, "QApplication", lambda _argv: qapp)
    monkeypatch.setattr(qapp, "exec", lambda: 0)
    monkeypatch.setattr(app_main.HotkeyFilter, "register", lambda _self, _spec: True)
    monkeypatch.setattr(QSystemTrayIcon, "show", lambda _self: None)
    monkeypatch.setattr(app_main, "is_frozen", lambda: True)
    monkeypatch.setattr(app_main, "start_watcher", lambda: started.append("watcher"))
    monkeypatch.setattr(app_main.UpdateScheduler, "start", lambda _self: started.append("updates"))
    assert app_main.main([]) == 0
    assert started == ["updates", "watcher"]


# ------------------------------------------------------------------ nothing blocks the UI thread
def test_the_environment_probe_runs_in_the_background(qapp: QApplication, harness: Harness) -> None:
    import threading

    controller = controller_for(harness)
    threads: list[str] = []
    original = controller._flow.probe

    def spy() -> object:
        threads.append(threading.current_thread().name)
        return original()

    controller._flow.probe = spy  # type: ignore[method-assign]
    probes: list[object] = []
    controller.probed.connect(probes.append)
    assert controller.environment is None
    assert controller.preview() is None  # nothing is known yet; it does not block to find out
    controller.probe_async()
    wait_for(qapp, lambda: bool(probes))
    assert threads
    assert threads[0] != threading.main_thread().name
    assert controller.environment is probes[0]
    assert controller.preview() is not None


def test_previews_for_other_choices_reuse_the_probe_and_do_no_io(
    qapp: QApplication, harness: Harness
) -> None:
    controller = controller_for(harness)
    probes: list[object] = []
    controller.probed.connect(probes.append)
    controller.probe_async()
    wait_for(qapp, lambda: bool(probes))

    def forbidden(*_a: object) -> object:
        raise AssertionError("a preview must not touch the network or the disk again")

    controller._flow.probe = forbidden  # type: ignore[method-assign]
    harness.system.free_mb = 1  # the disk filling up later changes nothing until a new probe
    first = controller.preview(SetupChoices(chat="llama3.2"))
    second = controller.preview(SetupChoices(chat="qwen3.5:9b"))
    assert first is not None
    assert second is not None
    assert first.download_mb != second.download_mb  # still a real calculation per choice


def test_a_failed_probe_degrades_to_ollama_missing(qapp: QApplication, harness: Harness) -> None:
    controller = controller_for(harness)

    def broken() -> object:
        raise OSError("network stack unavailable")

    controller._flow.probe = broken  # type: ignore[method-assign]
    probes: list[object] = []
    controller.probed.connect(probes.append)
    controller.probe_async()
    wait_for(qapp, lambda: bool(probes))
    assert controller.environment is not None
    assert controller.environment.ollama.value == "missing"


# ---------------------------------------------------------- cancelling, one run at a time
def slowed_pulls(harness: Harness) -> threading.Event:
    """Make model downloads wait until the returned event is set."""
    gate = threading.Event()
    original = harness.host.pull

    def slow_pull(model: str) -> object:
        gate.wait(5)
        return original(model)

    harness.host.pull = slow_pull  # type: ignore[method-assign,assignment]
    return gate


def test_closing_the_wizard_stops_a_running_setup(qapp: QApplication, harness: Harness) -> None:
    gate = slowed_pulls(harness)
    controller = controller_for(harness)
    cancelled: list[int] = []
    results: list[SetupResult] = []
    controller.cancelled.connect(lambda: cancelled.append(1))
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: controller.running)
    controller.cancel()
    gate.set()
    wait_for(qapp, lambda: bool(cancelled))
    assert cancelled == [1]
    assert results == []  # it did not carry on and finish
    assert "qwen3.5:9b" not in harness.host.pulled  # nothing after the cancel was started
    assert not controller.running
    assert harness.state.get_meta(SETUP_COMPLETED_KEY) is None


def test_cancelling_answers_a_pending_downgrade_question_with_no(
    qapp: QApplication, harness: Harness
) -> None:
    harness.model_rates = {"qwen3.5:9b": 2.0, "qwen3:8b": 25.0}
    controller = controller_for(harness)
    asked: list[SlowOffer] = []
    cancelled: list[int] = []
    results: list[SetupResult] = []
    controller.cancelled.connect(lambda: cancelled.append(1))
    controller.finished.connect(results.append)

    def question(offer: SlowOffer) -> None:
        asked.append(offer)
        controller.cancel()  # the user closes the wizard instead of answering

    controller.downgrade_offered.connect(question)
    controller.start(SetupOptions())
    wait_for(qapp, lambda: bool(cancelled) or bool(results))
    assert asked
    assert "qwen3:8b" not in harness.host.pulled  # the smaller model was not downloaded


def test_rejecting_the_wizard_cancels_the_controller(qapp: QApplication, harness: Harness) -> None:
    wizard = wizard_for(harness)
    stopped: list[int] = []
    wizard._controller.cancel = lambda: stopped.append(1)  # type: ignore[method-assign]
    wizard.reject()
    assert stopped == [1]


def test_a_second_start_while_running_is_refused(qapp: QApplication, harness: Harness) -> None:
    gate = slowed_pulls(harness)
    controller = controller_for(harness)
    results: list[SetupResult] = []
    controller.finished.connect(results.append)
    controller.start(SetupOptions())
    controller.start(SetupOptions())  # e.g. a retry click while the first run is still going
    gate.set()
    wait_for(qapp, lambda: bool(results))
    qapp.processEvents()
    assert len(results) == 1
    assert harness.host.pulled.count("qwen3-embedding:0.6b") == 1


def test_run_setup_wizard_opens_only_one_window(
    qapp: QApplication, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    class FakeWizard:
        def __init__(self, *_a: object, **_k: object) -> None:
            seen.append("created")

        def exec(self) -> int:
            # while the first wizard is open, the tray item is chosen again
            app_main.run_setup_wizard(Settings(), harness.state)
            return 0

        def raise_(self) -> None:
            seen.append("raised")

        def activateWindow(self) -> None:  # noqa: N802
            seen.append("activated")

    monkeypatch.setattr(app_main, "SetupWizard", FakeWizard)
    monkeypatch.setattr(app_main, "SetupController", lambda *_a, **_k: object())
    monkeypatch.setattr(app_main, "make_settings_controller", lambda *_a, **_k: object())
    monkeypatch.setattr(app_main, "load_catalog", lambda *_a: object())
    monkeypatch.setattr(app_main, "probe_hardware", object)
    app_main.run_setup_wizard(Settings(), harness.state)
    assert seen == ["created", "raised", "activated"]  # one window; the second call fronted it
    assert not app_main._open_wizards  # and the guard resets afterwards
