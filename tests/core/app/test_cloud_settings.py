"""Cloud providers in Settings: the controller methods, the Cloud tab and the add dialog."""

from types import SimpleNamespace

import openai
import pytest
from PySide6.QtWidgets import QApplication
from tests.core.app.test_models_panel import wait_for
from tests.core.conftest import Env
from tests.core.providers.test_openai_compat import KEY, FakeClient, api_error

from localdoc_finder.app.cloud_provider_dialog import CloudProviderDialog
from localdoc_finder.app.cloud_switcher import CloudChoice, CloudSwitcher
from localdoc_finder.app.cloud_tab import CloudTab
from localdoc_finder.app.settings_controller import CloudModel, SettingsController
from localdoc_finder.core.providers.openai_compat import OpenAICompatibleProvider
from localdoc_finder.core.secrets import MemoryKeyStore
from localdoc_finder.core.settings import CloudProviderSettings, SettingsError


@pytest.fixture
def client() -> FakeClient:
    fake = FakeClient()
    fake.model_entries = [
        SimpleNamespace(
            id="vendor/big",
            context_length=128000,
            pricing={"prompt": "0.00000015", "completion": "0.0000006"},
        ),
        SimpleNamespace(id="vendor/plain", context_length=32000),
        SimpleNamespace(id="vendor/zzz"),
    ]
    return fake


@pytest.fixture
def store() -> MemoryKeyStore:
    return MemoryKeyStore()


@pytest.fixture
def changes() -> list[int]:
    return []


@pytest.fixture
def controller(
    env: Env, store: MemoryKeyStore, client: FakeClient, changes: list[int]
) -> SettingsController:
    def factory(name: str, cfg: CloudProviderSettings, key: str) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(name, cfg, key, client=client)

    return SettingsController(
        env.data_dir / "settings.toml",
        env.state,
        store,
        make_provider=factory,
        on_changed=lambda: changes.append(1),
    )


# ------------------------------------------------------------------ controller
class TestProviders:
    def test_adding_saves_the_provider_and_the_key_and_activates_the_first(
        self, controller: SettingsController, store: MemoryKeyStore, changes: list[int]
    ) -> None:
        name = controller.add_provider("openrouter", "  sk-or-1  ")
        cloud = controller.settings().cloud
        provider = cloud.providers[name]
        assert name == "openrouter"
        assert (provider.base_url, provider.preset) == ("https://openrouter.ai/api/v1", name)
        assert provider.label == "OpenRouter"
        assert cloud.active == "openrouter"
        assert store.get("openrouter") == "sk-or-1"
        assert changes  # the context is rebuilt with the new provider

    def test_a_later_provider_does_not_steal_the_active_slot(
        self, controller: SettingsController
    ) -> None:
        controller.add_provider("openrouter", "k1")
        controller.add_provider("gemini", "k2")
        assert controller.settings().cloud.active == "openrouter"

    def test_adding_a_preset_again_updates_the_key_and_keeps_the_models(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openai", "old")
        controller.set_cloud_model("openai", "chat", "gpt-4o")
        controller.add_provider("openai", "new")
        assert store.get("openai") == "new"
        assert controller.settings().cloud.providers["openai"].models == {"chat": "gpt-4o"}

    def test_custom_needs_an_address_and_gets_a_free_name(
        self, controller: SettingsController
    ) -> None:
        with pytest.raises(SettingsError, match="base URL"):
            controller.add_provider("custom", "k")
        first = controller.add_provider("custom", "k", base_url="http://localhost:1234/v1")
        second = controller.add_provider("custom", "k", "http://other/v1", label="Lab box")
        assert (first, second) == ("custom", "custom-2")
        providers = controller.settings().cloud.providers
        assert providers[second].label == "Lab box"
        assert providers[first].label == ""

    def test_bad_input_is_refused_and_nothing_is_saved(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        with pytest.raises(SettingsError, match="API key"):
            controller.add_provider("openai", "   ")
        with pytest.raises(SettingsError, match="unknown provider"):
            controller.add_provider("skynet", "k")
        assert controller.settings().cloud.providers == {}
        assert store.get("openai") is None

    def test_removing_deletes_the_key_and_clears_the_active_slot(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "k1")
        controller.add_provider("openai", "k2")
        controller.remove_provider("openai")
        assert store.get("openai") is None
        assert "openai" not in controller.settings().cloud.providers
        assert controller.settings().cloud.active == "openrouter"  # untouched
        controller.remove_provider("openrouter")
        assert controller.settings().cloud.active is None
        with pytest.raises(SettingsError, match="no provider named"):
            controller.remove_provider("openrouter")

    def test_the_active_provider_can_be_switched(self, controller: SettingsController) -> None:
        controller.add_provider("openrouter", "k1")
        controller.add_provider("openai", "k2")
        controller.set_active_provider("openai")
        assert controller.settings().cloud.active == "openai"
        with pytest.raises(SettingsError, match="no provider named"):
            controller.set_active_provider("nope")

    def test_key_statuses_report_active_and_preset(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "k1")
        controller.add_provider("openai", "k2")
        store.delete("openai")
        rows = {s.provider: s for s in controller.key_statuses()}
        assert (rows["openrouter"].active, rows["openrouter"].preset) == (True, "openrouter")
        assert (rows["openai"].active, rows["openai"].has_key) == (False, False)


class TestModelChoices:
    def test_models_are_set_and_cleared_per_role(self, controller: SettingsController) -> None:
        controller.add_provider("openrouter", "k")
        controller.set_cloud_model("openrouter", "chat", "vendor/big")
        controller.set_cloud_model("openrouter", "match_scorer", "vendor/plain")
        models = controller.settings().cloud.providers["openrouter"].models
        assert models == {"chat": "vendor/big", "match_scorer": "vendor/plain"}
        controller.set_cloud_model("openrouter", "match_scorer", "")  # same as chat
        assert controller.settings().cloud.providers["openrouter"].models == {"chat": "vendor/big"}
        with pytest.raises(SettingsError, match="unknown role"):
            controller.set_cloud_model("openrouter", "poetry", "x")
        with pytest.raises(SettingsError, match="no provider named"):
            controller.set_cloud_model("nope", "chat", "x")

    def test_a_price_is_saved_per_model_and_must_not_be_negative(
        self, controller: SettingsController
    ) -> None:
        controller.add_provider("openai", "k")
        controller.set_model_price("openai", "gpt-4.1", 2.0, 8.0)
        assert controller.settings().cloud.providers["openai"].pricing == {"gpt-4.1": (2.0, 8.0)}
        with pytest.raises(SettingsError, match="negative"):
            controller.set_model_price("openai", "gpt-4.1", -1.0, 8.0)

    def test_favorites_toggle(self, controller: SettingsController) -> None:
        controller.add_provider("openai", "k")
        assert controller.toggle_favorite("openai", "gpt-4.1") is True
        assert controller.toggle_favorite("openai", "gpt-4o") is True
        assert controller.settings().cloud.providers["openai"].favorites == ["gpt-4.1", "gpt-4o"]
        assert controller.toggle_favorite("openai", "gpt-4.1") is False
        assert controller.settings().cloud.providers["openai"].favorites == ["gpt-4o"]
        controller.toggle_favorite("openai", "gpt-4o")
        assert controller.settings().cloud.providers["openai"].favorites == []


class TestRouting:
    @pytest.mark.parametrize("policy", ["auto", "cloud"])
    def test_ask_and_chat_routing_carries_code_questions_with_it(
        self, controller: SettingsController, policy: str, changes: list[int]
    ) -> None:
        controller.set_cloud_routing("chat", policy)
        routing = controller.settings().cloud.routing
        assert routing == {"chat": policy, "code_chat": policy}
        assert changes

    def test_match_routing_is_its_own(self, controller: SettingsController) -> None:
        controller.set_cloud_routing("match_scorer", "cloud")
        assert controller.settings().cloud.routing == {"match_scorer": "cloud"}

    def test_local_is_the_default_so_it_is_left_unset(self, controller: SettingsController) -> None:
        controller.set_cloud_routing("chat", "cloud")
        controller.set_cloud_routing("chat", "local")
        cloud = controller.settings().cloud
        assert cloud.routing == {}
        assert cloud.policy("chat") == cloud.policy("code_chat") == "local"

    def test_bad_roles_and_policies_are_refused(self, controller: SettingsController) -> None:
        with pytest.raises(SettingsError, match="unknown role"):
            controller.set_cloud_routing("poetry", "cloud")
        with pytest.raises(SettingsError, match="unknown routing"):
            controller.set_cloud_routing("chat", "sometimes")
        assert controller.settings().cloud.routing == {}

    def test_the_fallback_to_the_local_model_can_be_switched(
        self, controller: SettingsController
    ) -> None:
        assert controller.settings().cloud.fallback_to_local is True
        controller.set_fallback_to_local(False)
        assert controller.settings().cloud.fallback_to_local is False

    def test_forgetting_consent_calls_the_app(self, env: Env, store: MemoryKeyStore) -> None:
        forgotten: list[int] = []
        controller = SettingsController(
            env.data_dir / "settings.toml",
            env.state,
            store,
            forget_consent=lambda: forgotten.append(1),
        )
        controller.forget_cloud_consent()
        assert forgotten == [1]


class TestListing:
    def test_a_saved_provider_lists_models_with_context_and_price(
        self, controller: SettingsController
    ) -> None:
        controller.add_provider("openrouter", "k")
        models = controller.list_cloud_models("openrouter")
        assert models == [
            CloudModel("vendor/big", 128000, pytest.approx((0.15, 0.6))),  # type: ignore[arg-type]
            CloudModel("vendor/plain", 32000, None),
            CloudModel("vendor/zzz", None, None),
        ]

    def test_prices_the_user_entered_count(self, controller: SettingsController) -> None:
        controller.add_provider("openrouter", "k")
        controller.set_model_price("openrouter", "vendor/zzz", 1.0, 2.0)
        priced = {m.id: m.price for m in controller.list_cloud_models("openrouter")}
        assert priced["vendor/zzz"] == (1.0, 2.0)

    def test_listing_needs_a_stored_key(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "k")
        store.delete("openrouter")
        with pytest.raises(SettingsError, match="no API key"):
            controller.list_cloud_models("openrouter")
        with pytest.raises(SettingsError, match="no provider named"):
            controller.list_cloud_models("nope")

    def test_provider_errors_come_through_without_the_key(
        self, controller: SettingsController, client: FakeClient
    ) -> None:
        controller.add_provider("openrouter", KEY)
        client.list_error = api_error(openai.InternalServerError, 500, f"oops {KEY}")
        with pytest.raises(Exception, match=r"\[API KEY\]") as raised:
            controller.list_cloud_models("openrouter")
        assert KEY not in str(raised.value)

    def test_probing_uses_a_key_that_is_not_saved(
        self, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        models = controller.probe_provider("openrouter", "typed-key")
        assert [m.id for m in models] == ["vendor/big", "vendor/plain", "vendor/zzz"]
        assert controller.settings().cloud.providers == {}  # nothing saved
        assert store.get("openrouter") is None
        with pytest.raises(SettingsError, match="API key"):
            controller.probe_provider("openrouter", " ")
        with pytest.raises(SettingsError, match="base URL"):
            controller.probe_provider("custom", "k")


@pytest.mark.parametrize(
    ("model", "label"),
    [
        (CloudModel("m"), "m"),
        (CloudModel("m", 128000), "m · 128k ctx"),
        (CloudModel("m", 128000, (0.15, 0.6)), "m · 128k ctx · $0.15 / $0.60 per 1M"),
        (CloudModel("m", None, (0.075, 10.0)), "m · $0.075 / $10.00 per 1M"),
        (CloudModel("m", None, (0.0, 0.0)), "m · free"),
    ],
)
def test_model_labels_show_what_is_known(model: CloudModel, label: str) -> None:
    assert model.label == label


# ------------------------------------------------------------------ the tab
@pytest.fixture
def tab(qapp: QApplication, controller: SettingsController) -> CloudTab:
    return CloudTab(
        controller,
        add_dialog=lambda ctl, _parent: ctl.add_provider("openrouter", "sk-or"),
        confirm_remove=lambda _label: True,
    )


def listed(qapp: QApplication, tab: CloudTab) -> None:
    wait_for(qapp, lambda: tab._loading is None and tab.chat_picker.combo.count() > 0)


class TestCloudTab:
    def test_with_no_provider_everything_stays_local_and_no_model_pickers_show(
        self, tab: CloudTab
    ) -> None:
        assert tab.key_labels == []
        assert tab.models_box.isHidden()

    def test_adding_a_provider_shows_its_row_and_lists_its_models(
        self, qapp: QApplication, tab: CloudTab
    ) -> None:
        tab.add_provider.click()
        assert [label.text() for label in tab.key_labels] == ["OpenRouter: key stored"]
        assert tab.active_radios["openrouter"].isChecked()
        listed(qapp, tab)
        assert tab.chat_picker.combo.itemText(0) == "vendor/big · 128k ctx · $0.15 / $0.60 per 1M"
        assert tab.models_status.text() == "3 models"
        assert not tab.models_box.isHidden()

    def test_cancelling_the_dialog_changes_nothing(
        self, qapp: QApplication, controller: SettingsController
    ) -> None:
        tab = CloudTab(controller, add_dialog=lambda _ctl, _parent: None)
        tab.add_provider.click()
        assert tab.key_labels == []

    def test_choosing_a_model_saves_it_for_that_role(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        tab.chat_picker.combo.setCurrentIndex(1)
        tab.chat_picker.combo.activated.emit(1)
        assert tab.chat_picker.combo.currentText() == "vendor/plain"  # the id, not the description
        assert controller.settings().cloud.providers["openrouter"].models == {
            "chat": "vendor/plain"
        }
        line = tab.match_picker.combo.lineEdit()
        assert line is not None
        tab.match_picker.combo.setEditText("some/typed-model")  # any id may be typed
        line.editingFinished.emit()
        models = controller.settings().cloud.providers["openrouter"].models
        assert models["match_scorer"] == "some/typed-model"
        tab.match_picker.combo.setEditText("")  # back to "same as chat"
        line.editingFinished.emit()
        assert "match_scorer" not in controller.settings().cloud.providers["openrouter"].models

    def test_the_completer_matches_anywhere_in_the_name(self, tab: CloudTab) -> None:
        completer = tab.chat_picker.combo.completer()
        assert completer is not None
        assert completer.filterMode().name == "MatchContains"
        assert completer.caseSensitivity().name == "CaseInsensitive"

    def test_starred_models_move_to_the_top(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        tab.chat_picker.combo.setEditText("vendor/zzz")
        tab.chat_picker.star.click()
        assert controller.settings().cloud.providers["openrouter"].favorites == ["vendor/zzz"]
        assert tab.chat_picker.combo.itemText(0).startswith("★ vendor/zzz")
        assert tab.chat_picker.star.text() == "★"

    def test_an_unpriced_model_asks_for_a_price_only_while_the_budget_is_on(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        controller.set_cloud_model("openrouter", "chat", "vendor/zzz")
        tab.refresh()
        assert tab.chat_picker.price_row.isHidden()  # no budget: nothing to enforce
        tab.limit.click()
        assert not tab.chat_picker.price_row.isHidden()
        tab.chat_picker.price_in.setValue(1.5)
        tab.chat_picker.price_out.setValue(6.0)
        tab.chat_picker.save_price.click()
        pricing = controller.settings().cloud.providers["openrouter"].pricing
        assert pricing == {"vendor/zzz": (1.5, 6.0)}
        assert tab.chat_picker.price_row.isHidden()

    def test_a_priced_model_never_asks(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        controller.set_cloud_model("openrouter", "chat", "vendor/big")
        tab.limit.click()
        tab.refresh()
        assert tab.chat_picker.price_row.isHidden()

    def test_a_failed_listing_is_shown_and_not_retried_forever(
        self, qapp: QApplication, tab: CloudTab, client: FakeClient
    ) -> None:
        client.list_error = api_error(openai.AuthenticationError, 401)
        messages: list[str] = []
        tab.message.connect(messages.append)
        tab.add_provider.click()
        wait_for(qapp, lambda: tab._loading is None and "rejected" in tab.models_status.text())
        assert "rejected" in tab.models_status.text()
        assert tab.refresh_models.isEnabled()
        assert any("rejected" in m for m in messages)

    def test_refresh_lists_again(
        self, qapp: QApplication, tab: CloudTab, client: FakeClient
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        client.model_entries = [SimpleNamespace(id="vendor/new")]
        tab.refresh_models.click()
        wait_for(qapp, lambda: tab.chat_picker.combo.count() == 1)
        assert tab.chat_picker.combo.itemText(0) == "vendor/new"

    def test_switching_the_active_provider_lists_its_models(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        listed(qapp, tab)
        controller.add_provider("openai", "k2")
        controller.set_cloud_model("openai", "chat", "gpt-4.1")
        tab.refresh()
        tab.active_radios["openai"].click()
        assert controller.settings().cloud.active == "openai"
        assert tab.models_title.text() == "Models for OpenAI"
        assert tab.chat_picker.current_id() == "gpt-4.1"

    def test_set_key_stores_it_and_removing_forgets_everything(
        self, qapp: QApplication, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "old")
        tab = CloudTab(controller, ask_key=lambda _p: "sk-new", confirm_remove=lambda _label: True)
        tab.key_buttons[0].click()
        assert store.get("openrouter") == "sk-new"
        assert "sk-new" not in tab.key_labels[0].text()
        tab.remove_buttons[0].click()
        assert store.get("openrouter") is None
        assert tab.key_labels == []
        assert controller.settings().cloud.providers == {}

    def test_removing_can_be_cancelled(
        self, qapp: QApplication, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "old")
        tab = CloudTab(controller, confirm_remove=lambda _label: False)
        tab.remove_buttons[0].click()
        assert store.get("openrouter") == "old"
        assert len(tab.key_labels) == 1

    def test_without_a_key_it_says_so_instead_of_calling_the_network(
        self, qapp: QApplication, controller: SettingsController, store: MemoryKeyStore
    ) -> None:
        controller.add_provider("openrouter", "k")
        store.delete("openrouter")
        tab = CloudTab(controller)
        tab.load_models()
        assert tab.models_status.text() == "Set the API key to list the models."
        assert tab._loading is None


class TestRoutingControls:
    def test_the_boxes_show_the_saved_routing(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        controller.set_cloud_routing("chat", "auto")
        controller.set_cloud_routing("match_scorer", "cloud")
        tab.refresh()
        assert tab.routing.chat.currentData() == "auto"
        assert tab.routing.match.currentData() == "cloud"

    def test_the_default_is_only_when_i_press_answer_better(
        self, qapp: QApplication, tab: CloudTab
    ) -> None:
        tab.add_provider.click()
        assert tab.routing.chat.currentData() == "local"
        assert tab.routing.chat.currentText() == "Only when I press Answer better"
        assert tab.routing.match.currentData() == "local"

    def test_choosing_saves_for_that_feature(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        tab.routing.chat.setCurrentIndex(tab.routing.chat.findData("cloud"))
        tab.routing.chat.activated.emit(tab.routing.chat.currentIndex())
        assert controller.settings().cloud.routing == {"chat": "cloud", "code_chat": "cloud"}
        tab.routing.match.setCurrentIndex(tab.routing.match.findData("auto"))
        tab.routing.match.activated.emit(tab.routing.match.currentIndex())
        assert controller.settings().cloud.policy("match_scorer") == "auto"

    def test_the_fallback_checkbox_shows_and_saves_the_setting(
        self, qapp: QApplication, tab: CloudTab, controller: SettingsController
    ) -> None:
        tab.add_provider.click()
        assert tab.routing.fallback.isChecked()  # on by default
        tab.routing.fallback.click()
        assert controller.settings().cloud.fallback_to_local is False
        tab.refresh()
        assert not tab.routing.fallback.isChecked()

    def test_forget_withdraws_dont_ask_again(
        self, qapp: QApplication, env: Env, store: MemoryKeyStore
    ) -> None:
        forgotten: list[int] = []
        controller = SettingsController(
            env.data_dir / "settings.toml",
            env.state,
            store,
            forget_consent=lambda: forgotten.append(1),
        )
        controller.add_provider("openrouter", "k")
        tab = CloudTab(controller)
        messages: list[str] = []
        tab.message.connect(messages.append)
        tab.routing.forget.click()
        assert forgotten == [1]
        assert messages == ["will ask before sending"]


# ------------------------------------------------------------------ the popup switcher
@pytest.fixture
def switcher(controller: SettingsController) -> CloudSwitcher:
    return CloudSwitcher(controller)


def test_no_provider_means_no_choices(switcher: CloudSwitcher) -> None:
    assert switcher.choices() == []


def test_a_provider_offers_its_chat_model_and_its_favorites(
    controller: SettingsController,
    switcher: CloudSwitcher,
) -> None:
    controller.add_provider("openrouter", "k")
    controller.set_cloud_model("openrouter", "chat", "vendor/big")
    controller.toggle_favorite("openrouter", "vendor/plain")
    controller.toggle_favorite("openrouter", "vendor/big")  # already the chat model: listed once
    assert switcher.choices() == [
        CloudChoice("openrouter", "OpenRouter", "vendor/big", current=True),
        CloudChoice("openrouter", "OpenRouter", "vendor/plain", current=False),
    ]
    assert switcher.choices()[0].text == "OpenRouter / vendor/big"


def test_the_active_provider_comes_first_and_only_its_chat_model_is_current(
    controller: SettingsController,
    switcher: CloudSwitcher,
) -> None:
    controller.add_provider("openrouter", "k1")
    controller.add_provider("openai", "k2")
    controller.set_cloud_model("openrouter", "chat", "vendor/big")
    controller.set_cloud_model("openai", "chat", "gpt-4.1")
    controller.set_active_provider("openai")
    choices = switcher.choices()
    assert [(c.provider, c.current) for c in choices] == [("openai", True), ("openrouter", False)]


def test_a_provider_without_a_key_or_a_model_is_left_out(
    controller: SettingsController,
    switcher: CloudSwitcher,
    store: MemoryKeyStore,
) -> None:
    controller.add_provider("openrouter", "k1")
    controller.add_provider("openai", "k2")
    controller.set_cloud_model("openrouter", "chat", "vendor/big")
    controller.set_cloud_model("openai", "chat", "gpt-4.1")
    store.delete("openai")  # no key: a request to it could not be sent
    controller.add_provider("gemini", "k3")  # a key, but no model chosen yet
    assert [c.provider for c in switcher.choices()] == ["openrouter"]


def test_picking_sets_the_chat_model_and_the_active_provider(
    controller: SettingsController,
    switcher: CloudSwitcher,
) -> None:
    controller.add_provider("openrouter", "k1")
    controller.add_provider("openai", "k2")
    controller.set_cloud_model("openai", "chat", "gpt-4.1")
    controller.toggle_favorite("openai", "gpt-4o")
    switcher.choose(CloudChoice("openai", "OpenAI", "gpt-4o"))
    cloud = controller.settings().cloud
    assert cloud.active == "openai"
    assert cloud.providers["openai"].models["chat"] == "gpt-4o"
    # gpt-4o is now the chat model (and also a favorite: listed once); the old one is gone
    assert switcher.choices() == [CloudChoice("openai", "OpenAI", "gpt-4o", current=True)]


def test_picking_invalidates_the_context(
    env: Env,
    store: MemoryKeyStore,
) -> None:
    changes: list[int] = []
    controller = SettingsController(
        env.data_dir / "settings.toml", env.state, store, on_changed=lambda: changes.append(1)
    )
    controller.add_provider("openrouter", "k")
    changes.clear()
    CloudSwitcher(controller).choose(CloudChoice("openrouter", "OpenRouter", "vendor/big"))
    assert changes  # the app rebuilds its context from the new settings


# ------------------------------------------------------------------ the add dialog
@pytest.fixture
def opened() -> list[str]:
    return []


@pytest.fixture
def dialog(
    qapp: QApplication, controller: SettingsController, opened: list[str]
) -> CloudProviderDialog:
    return CloudProviderDialog(controller, open_url=opened.append)


class TestAddDialog:
    def test_the_key_page_link_opens_the_presets_page(
        self, dialog: CloudProviderDialog, opened: list[str]
    ) -> None:
        dialog.preset.setCurrentIndex(dialog.preset.findData("gemini"))
        dialog.key_link.click()
        assert opened == ["https://aistudio.google.com/apikey"]

    def test_only_custom_asks_for_an_address_and_has_no_key_page(
        self, dialog: CloudProviderDialog
    ) -> None:
        assert dialog.base_url.isHidden()
        dialog.preset.setCurrentIndex(dialog.preset.findData("custom"))
        assert not dialog.base_url.isHidden()
        assert dialog.key_link.isHidden()
        dialog.preset.setCurrentIndex(dialog.preset.findData("openai"))
        assert dialog.base_url.isHidden()
        assert not dialog.key_link.isHidden()

    def test_the_key_field_hides_what_is_typed(self, dialog: CloudProviderDialog) -> None:
        assert dialog.key.echoMode().name == "Password"

    def test_a_good_key_saves_and_closes(
        self,
        qapp: QApplication,
        dialog: CloudProviderDialog,
        controller: SettingsController,
        store: MemoryKeyStore,
    ) -> None:
        dialog.preset.setCurrentIndex(dialog.preset.findData("openrouter"))
        dialog.key.setText("sk-or-typed")
        dialog.test.click()
        assert not dialog.test.isEnabled()  # busy while it tests
        wait_for(qapp, lambda: dialog.added is not None)
        assert dialog.added == "openrouter"
        assert store.get("openrouter") == "sk-or-typed"
        assert controller.settings().cloud.active == "openrouter"
        assert dialog.key.text() == ""  # not left in the field

    def test_a_bad_key_shows_the_error_and_saves_nothing(
        self,
        qapp: QApplication,
        dialog: CloudProviderDialog,
        controller: SettingsController,
        store: MemoryKeyStore,
        client: FakeClient,
    ) -> None:
        client.list_error = api_error(openai.AuthenticationError, 401, f"bad {KEY}")
        dialog.key.setText(KEY)
        dialog.test.click()
        wait_for(qapp, lambda: "rejected" in dialog.status.text())
        assert "rejected" in dialog.status.text()
        assert KEY not in dialog.status.text()
        assert dialog.added is None
        assert dialog.test.isEnabled()
        assert controller.settings().cloud.providers == {}
        assert store.get("openrouter") is None

    def test_an_empty_key_is_refused_before_any_call(
        self, qapp: QApplication, dialog: CloudProviderDialog, client: FakeClient
    ) -> None:
        dialog.test.click()
        wait_for(qapp, lambda: dialog.status.text() != "Testing the key and listing models…")
        assert "API key" in dialog.status.text()
        assert client.completions.calls == []

    def test_a_test_finishing_after_the_dialog_was_closed_saves_nothing(
        self,
        qapp: QApplication,
        dialog: CloudProviderDialog,
        controller: SettingsController,
        store: MemoryKeyStore,
    ) -> None:
        dialog.key.setText("sk-late")
        dialog.test.click()
        dialog.reject()
        for _ in range(50):
            qapp.processEvents()
        assert dialog.added is None
        assert controller.settings().cloud.providers == {}
        assert store.get("openrouter") is None
