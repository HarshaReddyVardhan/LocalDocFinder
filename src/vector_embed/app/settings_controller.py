"""Qt-free logic behind the Settings window: every change goes through ``set_setting``."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from vector_embed.app.hotkey import parse_hotkey
from vector_embed.core.lifecycle import schedule_data_deletion, stop_other_instances
from vector_embed.core.models.benchmark import BenchResult, load_results
from vector_embed.core.secrets import KeyStore
from vector_embed.core.settings import Settings, SettingsError, load_settings
from vector_embed.core.settings_io import set_setting
from vector_embed.core.store.sqlite import StateDb
from vector_embed.core.updates import UpdateKind, UpdateOutcome, Updater

PACKAGE_NAME = "vector-embed"
NO_UPDATES = "Updates are not available in this build."


@dataclass(frozen=True)
class KeyStatus:
    provider: str
    label: str
    has_key: bool


def app_version() -> str:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "development"


class SettingsController:
    def __init__(
        self,
        settings_path: Path,
        state: StateDb,
        keys: KeyStore,
        *,
        apply_autostart: Callable[[bool], None] = lambda enabled: None,
        updater: Updater | None = None,
        stop_others: Callable[[], object] = stop_other_instances,
        schedule_deletion: Callable[[Path], None] = schedule_data_deletion,
    ) -> None:
        self._path = settings_path
        self._state = state
        self._keys = keys
        self._apply_autostart = apply_autostart
        self._updater = updater
        self._stop_others = stop_others
        self._schedule_deletion = schedule_deletion

    @property
    def settings_path(self) -> Path:
        return self._path

    def settings(self) -> Settings:
        """The current settings, re-read from disk so the window never shows stale values."""
        return load_settings(self._path)

    # ------------------------------------------------------------------ general
    def set_hotkey(self, spec: str) -> str:
        """Validate and save a hotkey such as ``ctrl+alt+space``; returns the normalised text."""
        cleaned = spec.strip().lower().replace(" ", "")
        try:
            parse_hotkey(cleaned)
        except ValueError as exc:
            raise SettingsError(f"invalid hotkey {spec!r}: {exc}") from exc
        set_setting(self._path, ["search", "hotkey"], cleaned)
        return cleaned

    def set_roots(self, roots: list[str]) -> None:
        if not roots:
            raise SettingsError("at least one folder must be indexed")
        set_setting(self._path, ["scope", "roots"], list(dict.fromkeys(roots)))

    def set_start_with_windows(self, enabled: bool) -> None:
        set_setting(self._path, ["app", "start_with_windows"], enabled)
        self._apply_autostart(enabled)

    # ------------------------------------------------------------------ models
    def speed_tests(self) -> list[BenchResult]:
        return load_results(self._state)

    # ------------------------------------------------------------------ cloud and privacy
    def key_statuses(self) -> list[KeyStatus]:
        providers = self.settings().cloud.providers
        return [
            KeyStatus(name, cfg.label or name, self._keys.get(name) is not None)
            for name, cfg in providers.items()
        ]

    def set_key(self, provider: str, key: str) -> None:
        self._keys.set(provider, key)

    def delete_key(self, provider: str) -> None:
        self._keys.delete(provider)

    def set_redact_personal(self, enabled: bool) -> None:
        set_setting(self._path, ["privacy", "redact_personal"], enabled)

    def set_mask_ids_locally(self, enabled: bool) -> None:
        set_setting(self._path, ["privacy", "mask_ids_locally"], enabled)

    def set_monthly_budget(self, usd: float | None) -> None:
        if usd is not None and usd <= 0:
            raise SettingsError("the monthly budget must be above zero (or off)")
        set_setting(self._path, ["cloud", "monthly_budget_usd"], usd)

    # ------------------------------------------------------------------ updates
    def set_auto_check(self, enabled: bool) -> None:
        set_setting(self._path, ["updates", "auto_check"], enabled)

    def check_now(self) -> UpdateOutcome:
        """Look for an update and download it; may take a while, so call it off the UI thread."""
        if self._updater is None:
            return UpdateOutcome(UpdateKind.NOT_CONFIGURED, NO_UPDATES)
        return self._updater.check()

    def restart_to_update(self) -> None:
        if self._updater is None:
            raise RuntimeError(NO_UPDATES)
        self._updater.restart_to_update()

    # ------------------------------------------------------------------ data
    def delete_my_data(self) -> None:
        """Stop everything, then remove the data folder once this process has exited."""
        self._apply_autostart(False)  # so nothing restarts at the next logon
        self._stop_others()
        self._schedule_deletion(self._path.parent)
