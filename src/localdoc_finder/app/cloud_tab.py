"""Settings -> Cloud & Privacy: add providers, set keys, choose models, privacy and budget."""

from collections.abc import Callable

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from localdoc_finder.app.cloud_model_picker import ModelPicker
from localdoc_finder.app.cloud_provider_dialog import (
    ModelsJob,
    ModelsResult,
    ModelsSignals,
    run_add_provider_dialog,
)
from localdoc_finder.app.cloud_routing_box import RoutingBox
from localdoc_finder.app.settings_controller import CloudModel, SettingsController
from localdoc_finder.app.settings_tabs import SettingsTab, ask_secret
from localdoc_finder.app.theme import scheme_in_use, style_check_boxes
from localdoc_finder.core.models.catalog import ROLE_CHAT, ROLE_MATCH_SCORER
from localdoc_finder.core.settings import CloudProviderSettings

DEFAULT_BUDGET_USD = 10.0
NO_PROVIDER_TEXT = "No cloud provider is configured; everything stays local."


def confirm_remove_dialog(label: str) -> bool:
    answer = QMessageBox.question(
        None,
        "Remove provider",
        f"Remove {label} and its stored API key?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return answer == QMessageBox.StandardButton.Yes


class CloudTab(SettingsTab):
    def __init__(
        self,
        controller: SettingsController,
        ask_key: Callable[[str], str | None] = ask_secret,
        add_dialog: Callable[
            [SettingsController, QWidget | None], str | None
        ] = run_add_provider_dialog,
        confirm_remove: Callable[[str], bool] = confirm_remove_dialog,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(controller)
        self._ask_key = ask_key
        self._add_dialog = add_dialog
        self._confirm_remove = confirm_remove
        self._pool = pool or QThreadPool.globalInstance()
        self._signals = ModelsSignals()
        self._signals.done.connect(self._models_listed)
        self._models: dict[str, list[CloudModel]] = {}  # per provider, from the last listing
        self._loading: str | None = None
        self._active: str | None = None

        self.providers = QVBoxLayout()
        self.key_labels: list[QLabel] = []
        self.key_buttons: list[QPushButton] = []
        self.remove_buttons: list[QPushButton] = []
        self.active_radios: dict[str, QRadioButton] = {}
        self._radio_group = QButtonGroup(self)
        self.add_provider = QPushButton("+ Add provider…")
        self._build_models_box()
        self._build_privacy_box()

        add_row = QHBoxLayout()
        add_row.addWidget(self.add_provider)
        add_row.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Cloud providers"))
        layout.addLayout(self.providers)
        layout.addLayout(add_row)
        layout.addWidget(self.models_box)
        layout.addWidget(self.privacy_box)
        layout.addStretch(1)
        self.refresh()
        self._connect()

    def _build_privacy_box(self) -> None:
        self.redact = QCheckBox("Hide my name, email and phone from cloud models")
        self.mask_local = QCheckBox("Also mask ID numbers for local models")
        self.limit = QCheckBox("Limit monthly cloud spend")
        self.budget = QDoubleSpinBox()
        self.budget.setPrefix("$ ")
        self.budget.setRange(1.0, 100000.0)
        budget_row = QHBoxLayout()
        budget_row.addWidget(self.limit)
        budget_row.addWidget(self.budget)
        budget_row.addStretch(1)
        self.privacy_box = QWidget()
        privacy = QVBoxLayout(self.privacy_box)
        privacy.setContentsMargins(0, 0, 0, 0)
        privacy.addWidget(QLabel("Privacy"))
        privacy.addWidget(self.redact)
        privacy.addWidget(self.mask_local)
        privacy.addLayout(budget_row)

    def _connect(self) -> None:
        self.redact.clicked.connect(lambda on: self._save(self._controller.set_redact_personal, on))
        self.mask_local.clicked.connect(
            lambda on: self._save(self._controller.set_mask_ids_locally, on)
        )
        self.limit.clicked.connect(lambda _on: self._save_budget())
        self.budget.editingFinished.connect(self._save_budget)
        self.add_provider.clicked.connect(self._add)
        self.refresh_models.clicked.connect(lambda: self.load_models(force=True))
        self.routing.routing_changed.connect(self._route)
        self.routing.forget_clicked.connect(
            lambda: self._guard(self._controller.forget_cloud_consent, "will ask before sending")
        )
        for role, picker in self._pickers():
            picker.chosen.connect(lambda model, r=role: self._choose_model(r, model))
            picker.star_toggled.connect(self._star)
            picker.price_entered.connect(self._set_price)

    def _build_models_box(self) -> None:
        self.models_title = QLabel("")
        self.chat_picker = ModelPicker("Ask & Chat model")
        self.match_picker = ModelPicker("Match model", "Same as Ask & Chat")
        self.refresh_models = QPushButton("Refresh list")
        self.models_status = QLabel("")
        self.models_status.setWordWrap(True)
        self.models_box = QWidget()
        models = QVBoxLayout(self.models_box)
        models.setContentsMargins(0, 0, 0, 0)
        models.addWidget(self.models_title)
        models.addWidget(self.chat_picker)
        models.addWidget(self.match_picker)
        refresh_row = QHBoxLayout()
        refresh_row.addWidget(self.refresh_models)
        refresh_row.addWidget(self.models_status, 1)
        models.addLayout(refresh_row)
        self.routing = RoutingBox()
        models.addWidget(self.routing)

    def _pickers(self) -> tuple[tuple[str, ModelPicker], tuple[str, ModelPicker]]:
        return (ROLE_CHAT, self.chat_picker), (ROLE_MATCH_SCORER, self.match_picker)

    # ------------------------------------------------------------------ drawing
    def refresh(self) -> None:
        settings = self._controller.settings()
        self.redact.setChecked(settings.privacy.redact_personal)
        self.mask_local.setChecked(settings.privacy.mask_ids_locally)
        limit = settings.cloud.monthly_budget_usd
        self.limit.setChecked(limit is not None)
        self.budget.setValue(limit if limit is not None else DEFAULT_BUDGET_USD)
        self.budget.setEnabled(limit is not None)
        self._fill_providers()
        self._fill_models()

    def _fill_providers(self) -> None:
        while self.providers.count():
            item = self.providers.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        for radio in self._radio_group.buttons():
            self._radio_group.removeButton(radio)
        self.key_labels, self.key_buttons, self.remove_buttons = [], [], []
        self.active_radios = {}
        statuses = self._controller.key_statuses()
        self._active = next((s.provider for s in statuses if s.active), None)
        if not statuses:
            self.providers.addWidget(QLabel(NO_PROVIDER_TEXT))
        for status in statuses:
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"{status.label}: {'key stored' if status.has_key else 'no key'}")
            radio = QRadioButton("Active")
            radio.setChecked(status.active)
            set_key = QPushButton("Set key…")
            remove = QPushButton("Remove")
            self._radio_group.addButton(radio)
            self.key_labels.append(label)
            self.key_buttons.append(set_key)
            self.remove_buttons.append(remove)
            self.active_radios[status.provider] = radio
            radio.clicked.connect(lambda _c=False, name=status.provider: self._activate(name))
            set_key.clicked.connect(lambda _c=False, name=status.provider: self._set_key(name))
            remove.clicked.connect(
                lambda _c=False, name=status.provider, text=status.label: self._remove(name, text)
            )
            for widget in (label, radio, set_key, remove):
                line.addWidget(widget, 1 if widget is label else 0)
            style_check_boxes(row, scheme_in_use())  # radios made after the window was styled
            self.providers.addWidget(row)

    def _active_settings(self) -> CloudProviderSettings | None:
        if self._active is None:
            return None
        return self._controller.settings().cloud.providers.get(self._active)

    def _fill_models(self) -> None:
        provider = self._active_settings()
        self.models_box.setVisible(provider is not None)
        if provider is None:
            return
        self.models_title.setText(f"Models for {provider.label or self._active}")
        cloud = self._controller.settings().cloud
        self.routing.set_policies(cloud.policy(ROLE_CHAT), cloud.policy(ROLE_MATCH_SCORER))
        for role, picker in self._pickers():
            picker.set_current(provider.models.get(role, ""))
        self._fill_lists()
        if not self.models_status.text():
            has_models = bool(self._models.get(self._active or ""))
            self.models_status.setText(
                "" if has_models else "Press Refresh list to see the models."
            )

    def _fill_lists(self) -> None:
        """Put the listed models and favourites in the pickers, keeping what is in their boxes."""
        provider = self._active_settings()
        if provider is None:
            return
        models = self._models.get(self._active or "", [])
        for _role, picker in self._pickers():
            picker.set_models(models, provider.favorites)
        self._update_price_rows()

    def _update_price_rows(self) -> None:
        """Ask for a price only when the budget is on and the model's cost is not known."""
        provider = self._active_settings()
        budget_on = self.limit.isChecked()
        for _role, picker in self._pickers():
            model = picker.current_id()
            unknown = provider is not None and model and not self._priced(provider, model)
            picker.show_price_entry(budget_on and bool(unknown))

    def _priced(self, provider: CloudProviderSettings, model: str) -> bool:
        if model in provider.pricing:
            return True
        return any(
            m.id == model and m.price is not None for m in self._models.get(self._active or "", [])
        )

    # ------------------------------------------------------------------ model listing
    def showEvent(self, event: object) -> None:  # noqa: N802
        super().showEvent(event)  # type: ignore[arg-type]
        self.load_models()  # once per provider; the Refresh button forces a new look

    def load_models(self, *, force: bool = False) -> None:
        """List the active provider's models in the background (nothing is sent but the key)."""
        name = self._active
        if name is None or self._loading is not None:
            return
        if not force and name in self._models:
            return
        if not any(s.has_key for s in self._controller.key_statuses() if s.provider == name):
            self.models_status.setText("Set the API key to list the models.")
            return
        self._loading = name
        self.refresh_models.setEnabled(False)
        self.models_status.setText("Loading models…")
        job = ModelsJob(name, lambda: self._controller.list_cloud_models(name), self._signals)
        self._pool.start(job)

    def _models_listed(self, result: ModelsResult) -> None:
        self._loading = None
        self.refresh_models.setEnabled(True)
        if result.error:
            self.models_status.setText(result.error)
            self.message.emit(result.error)
            return
        self._models[result.owner] = result.models
        self.models_status.setText(f"{len(result.models)} models")
        if result.owner == self._active:
            self._fill_lists()
        else:  # the user switched provider meanwhile: list the one now active
            self.load_models()

    # ------------------------------------------------------------------ actions
    def _add(self) -> None:
        name = self._add_dialog(self._controller, self)
        if name is None:
            return
        self._models.pop(name, None)
        self.message.emit("provider added")
        self.refresh()
        self.load_models()

    def _set_key(self, provider: str) -> None:
        key = self._ask_key(provider)
        if key and self._guard(lambda: self._controller.set_key(provider, key), "key saved"):
            self._models.pop(provider, None)
            self._fill_providers()
            self.load_models()

    def _remove(self, name: str, label: str) -> None:
        if self._confirm_remove(label) and self._guard(
            lambda: self._controller.remove_provider(name), "provider removed"
        ):
            self._models.pop(name, None)
            self.refresh()

    def _activate(self, name: str) -> None:
        if self._guard(lambda: self._controller.set_active_provider(name)):
            self.models_status.setText("")
            self.refresh()
            self.load_models()
        else:
            self.refresh()

    def _choose_model(self, role: str, model: str) -> None:
        name = self._active
        if name is None:
            return
        self._guard(lambda: self._controller.set_cloud_model(name, role, model), "model saved")
        self._update_price_rows()

    def _route(self, role: str, policy: str) -> None:
        saved = self._guard(
            lambda: self._controller.set_cloud_routing(role, policy), "routing saved"
        )
        if not saved:
            self._fill_models()

    def _star(self, model: str) -> None:
        name = self._active
        if name is not None and self._guard(lambda: self._controller.toggle_favorite(name, model)):
            self._fill_lists()

    def _set_price(self, model: str, inp: float, out: float) -> None:
        name = self._active
        if name is not None and self._guard(
            lambda: self._controller.set_model_price(name, model, inp, out), "price saved"
        ):
            self._update_price_rows()

    def _save(self, setter: Callable[[bool], None], value: bool) -> None:
        self._guard(lambda: setter(value), "saved")

    def _save_budget(self) -> None:
        self.budget.setEnabled(self.limit.isChecked())
        value = self.budget.value() if self.limit.isChecked() else None
        self._guard(lambda: self._controller.set_monthly_budget(value), "budget saved")
        self._update_price_rows()
