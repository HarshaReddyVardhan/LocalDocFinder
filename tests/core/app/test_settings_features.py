import pytest
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication
from tests.core.app.test_models_panel import wait_for
from tests.core.app.test_settings_window import FakeKeys
from tests.core.conftest import Env

from vector_embed.app.settings_controller import SettingsController
from vector_embed.app.settings_features import FeaturesTab
from vector_embed.core.features import enabled_features
from vector_embed.core.models.manager import ProgressCallback
from vector_embed.core.models.starter import StarterPick
from vector_embed.core.providers.base import ProviderError, PullProgress
from vector_embed.core.settings import SettingsError

PICK = StarterPick("chat", "qwen3.5:9b", "fits your 8192 MB of VRAM", 6100, "qwen3:8b")


class FakeModels:
    def __init__(self, missing: StarterPick | None = PICK) -> None:
        self.missing = missing
        self.pulled: list[str] = []
        self.fail: str | None = None

    def missing_chat_model(self) -> StarterPick | None:
        return self.missing

    def pull(self, name: str, progress: ProgressCallback | None = None) -> None:
        if self.fail:
            raise ProviderError(self.fail)
        if progress is not None:
            progress(PullProgress("pulling", 50, 100))
        self.pulled.append(name)
        self.missing = None


@pytest.fixture
def changed() -> list[int]:
    return []


@pytest.fixture
def controller(env: Env, changed: list[int]) -> SettingsController:
    return SettingsController(
        env.data_dir / "settings.toml",
        env.state,
        FakeKeys(),
        on_changed=lambda: changed.append(1),
    )


def test_set_feature_saves_and_applies(controller: SettingsController, changed: list[int]) -> None:
    controller.set_feature("chat", True)
    assert enabled_features(controller.settings()) == {"chat"}
    assert changed == [1]
    with pytest.raises(SettingsError, match="unknown feature"):
        controller.set_feature("search", False)


def test_the_tab_starts_search_only_with_no_download_offered(
    qapp: QApplication, controller: SettingsController
) -> None:
    tab = FeaturesTab(controller, FakeModels())
    assert not any(box.isChecked() for box in tab.boxes.values())
    assert tab.download_row.isHidden()


def test_switching_a_feature_on_offers_the_missing_chat_model(
    qapp: QApplication, controller: SettingsController
) -> None:
    models = FakeModels()
    tab = FeaturesTab(controller, models)
    messages: list[str] = []
    tab.message.connect(messages.append)
    tab.boxes["ask"].click()
    assert enabled_features(controller.settings()) == {"ask"}
    wait_for(qapp, lambda: not tab.download_row.isHidden())
    assert "qwen3.5:9b" in tab.model_note.text()
    assert "6.0 GB" in tab.model_note.text()
    tab.download.click()
    wait_for(qapp, lambda: bool(models.pulled))
    wait_for(qapp, tab.download_row.isHidden)
    assert models.pulled == ["qwen3.5:9b"]
    assert "ready" in messages[-1]


def test_no_download_is_offered_when_a_chat_model_is_installed(
    qapp: QApplication, controller: SettingsController
) -> None:
    tab = FeaturesTab(controller, FakeModels(missing=None))
    tab.boxes["match"].click()
    QThreadPool.globalInstance().waitForDone()
    qapp.processEvents()  # deliver the answer from the background check
    assert tab.download_row.isHidden()


def test_switching_every_feature_off_hides_the_offer(
    qapp: QApplication, controller: SettingsController
) -> None:
    tab = FeaturesTab(controller, FakeModels())
    tab.boxes["chat"].click()
    wait_for(qapp, lambda: not tab.download_row.isHidden())
    tab.boxes["chat"].click()
    assert tab.download_row.isHidden()
    assert enabled_features(controller.settings()) == frozenset()


def test_a_failed_download_can_be_retried(
    qapp: QApplication, controller: SettingsController
) -> None:
    models = FakeModels()
    models.fail = "network down"
    tab = FeaturesTab(controller, models)
    messages: list[str] = []
    tab.message.connect(messages.append)
    tab.boxes["ask"].click()
    wait_for(qapp, lambda: not tab.download_row.isHidden())
    tab.download.click()
    wait_for(qapp, lambda: "network down" in messages)
    assert tab.download.isEnabled()
    assert tab.progress.isHidden()
