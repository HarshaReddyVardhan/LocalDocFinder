"""Qt-free logic behind the Settings window: every change goes through ``set_setting``."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from vector_embed.app.hotkey import parse_hotkey
from vector_embed.app.theme import THEME_CHOICES
from vector_embed.core.features import FEATURES
from vector_embed.core.file_kinds import PRESET_CUSTOM, FileKind
from vector_embed.core.lifecycle import (
    ensure_data_folder,
    schedule_data_deletion,
    stop_everything,
)
from vector_embed.core.models.benchmark import BenchResult, load_results
from vector_embed.core.protection import SystemProtection
from vector_embed.core.secrets import KeyStore
from vector_embed.core.settings import Settings, SettingsError, load_settings
from vector_embed.core.settings_io import set_setting, set_settings
from vector_embed.core.settings_schema import OptionSpec
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
        stop_others: Callable[[], object] = stop_everything,
        schedule_deletion: Callable[[Path], None] = schedule_data_deletion,
        on_changed: Callable[[], None] = lambda: None,
    ) -> None:
        self._on_changed = on_changed
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
        self._on_changed()
        return cleaned

    def set_scope(
        self,
        coverage: str,
        roots: list[str],
        file_types: str,
        protection: SystemProtection | None = None,
        custom_kinds: Iterable[str] = (),
    ) -> None:
        """Save what to index: the whole PC or chosen folders/drives, and which file types.

        System folders are refused by name here (and skipped regardless in the scan).
        ``custom_kinds`` (``core.file_kinds.FileKind`` values) only counts for "custom".
        """
        protection = protection or SystemProtection()
        chosen = list(dict.fromkeys(roots))
        if coverage == "chosen" and not chosen:
            raise SettingsError("choose at least one folder or drive to index")
        for root in chosen:
            reason = protection.reason(root)
            if reason:
                raise SettingsError(f"{root} is {reason}; it is never indexed")
        kinds = sorted(set(custom_kinds)) if file_types == PRESET_CUSTOM else []
        if file_types == PRESET_CUSTOM and not kinds:
            raise SettingsError("choose at least one kind of file to index")
        unknown = [kind for kind in kinds if kind not in set(FileKind)]
        if unknown:
            raise SettingsError(f"unknown kind of file {unknown[0]!r}")
        set_settings(
            self._path,
            [
                (["scope", "coverage"], coverage),
                (["scope", "roots"], chosen or None),
                (["scope", "file_types"], file_types),
                (["scope", "custom_kinds"], kinds or None),
            ],
        )
        self._on_changed()

    def set_start_with_windows(self, enabled: bool) -> None:
        set_setting(self._path, ["app", "start_with_windows"], enabled)
        self._apply_autostart(enabled)
        self._on_changed()

    def set_theme(self, choice: str) -> None:
        if choice not in THEME_CHOICES:
            raise SettingsError(f"unknown theme {choice!r}")
        set_setting(self._path, ["app", "theme"], choice)
        self._on_changed()

    def set_feature(self, name: str, enabled: bool) -> None:
        """Switch Ask, Chat or Match on or off; Search is always on."""
        if name not in FEATURES:
            raise SettingsError(f"unknown feature {name!r}")
        set_setting(self._path, ["features", name], enabled)
        self._on_changed()

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
        self._on_changed()

    def delete_key(self, provider: str) -> None:
        self._keys.delete(provider)
        self._on_changed()

    def set_redact_personal(self, enabled: bool) -> None:
        set_setting(self._path, ["privacy", "redact_personal"], enabled)
        self._on_changed()

    def set_mask_ids_locally(self, enabled: bool) -> None:
        set_setting(self._path, ["privacy", "mask_ids_locally"], enabled)
        self._on_changed()

    def set_monthly_budget(self, usd: float | None) -> None:
        if usd is not None and usd <= 0:
            raise SettingsError("the monthly budget must be above zero (or off)")
        set_setting(self._path, ["cloud", "monthly_budget_usd"], usd)
        self._on_changed()

    # ------------------------------------------------------------------ advanced
    def set_option(self, option: OptionSpec, value: object) -> None:
        """Save one option of the generated Advanced page (validated against the full schema;
        ``None`` clears it back to the default)."""
        set_setting(self._path, list(option.path), value)
        self._on_changed()

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
        data_dir = self._state.data_dir  # the folder actually in use, not a guessed one
        ensure_data_folder(data_dir)  # before anything is stopped or removed
        self._apply_autostart(False)  # so nothing restarts at the next logon
        self._stop_others()
        for provider in self.settings().cloud.providers:
            self._keys.delete(provider)
        self._schedule_deletion(data_dir)
