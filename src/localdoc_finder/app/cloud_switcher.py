"""The popup's cloud model switcher: what to offer, and what picking one changes.

Qt-free, so the choices are testable. A choice is a model of a provider that has a key: the
provider's current chat model and the models the user starred.
"""

from dataclasses import dataclass

from localdoc_finder.app.settings_controller import SettingsController
from localdoc_finder.core.models.catalog import ROLE_CHAT


@dataclass(frozen=True)
class CloudChoice:
    provider: str
    provider_label: str
    model: str
    current: bool = False  # the active provider's chat model, i.e. where "Answer better" goes

    @property
    def text(self) -> str:
        return f"{self.provider_label} / {self.model}"


class CloudSwitcher:
    def __init__(self, controller: SettingsController) -> None:
        self._controller = controller

    def choices(self) -> list[CloudChoice]:
        """Every offered model, the active provider first. A provider without a key is left out:
        a request to it could not be sent."""
        providers = self._controller.settings().cloud.providers
        statuses = sorted(self._controller.key_statuses(), key=lambda s: not s.active)
        choices: list[CloudChoice] = []
        for status in statuses:
            provider = providers.get(status.provider)
            if provider is None or not status.has_key:
                continue
            chat_model = provider.models.get(ROLE_CHAT)
            for model in dict.fromkeys([m for m in (chat_model, *provider.favorites) if m]):
                current = status.active and model == chat_model
                choices.append(CloudChoice(status.provider, status.label, model, current))
        return choices

    def choose(self, choice: CloudChoice) -> None:
        """Make ``choice`` the model for Ask and Chat on its provider, and that provider active."""
        self._controller.set_cloud_model(choice.provider, ROLE_CHAT, choice.model)
        self._controller.set_active_provider(choice.provider)
