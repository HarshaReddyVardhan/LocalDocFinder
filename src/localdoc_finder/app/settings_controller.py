"""Qt-free logic behind the Settings window: every change goes through ``set_setting``."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from localdoc_finder.app.hotkey import parse_hotkey
from localdoc_finder.app.theme import THEME_CHOICES
from localdoc_finder.core.features import FEATURES
from localdoc_finder.core.file_kinds import PRESET_CUSTOM, FileKind
from localdoc_finder.core.lifecycle import (
    ensure_data_folder,
    schedule_data_deletion,
    stop_everything,
)
from localdoc_finder.core.models.benchmark import BenchResult, load_results
from localdoc_finder.core.models.catalog import (
    ROLE_CHAT,
    ROLE_CODE_CHAT,
    ROLE_MATCH_SCORER,
    ROLES,
)
from localdoc_finder.core.protection import SystemProtection
from localdoc_finder.core.providers.openai_compat import OpenAICompatibleProvider
from localdoc_finder.core.providers.presets import CUSTOM, PRESETS
from localdoc_finder.core.secrets import KeyStore
from localdoc_finder.core.settings import (
    CloudProviderSettings,
    Settings,
    SettingsError,
    load_settings,
)
from localdoc_finder.core.settings_io import set_setting, set_settings
from localdoc_finder.core.settings_schema import OptionSpec
from localdoc_finder.core.store.sqlite import StateDb
from localdoc_finder.core.updates import UpdateKind, UpdateOutcome, Updater

PACKAGE_NAME = "localdoc-finder"
NO_UPDATES = "Updates are not available in this build."
_TOKENS_PER_K = 1000
ROUTED_ROLES = (ROLE_CHAT, ROLE_MATCH_SCORER)  # what Settings lets the user point at the cloud
ROUTING_POLICIES = ("local", "auto", "cloud")
_CENTS_THRESHOLD_USD = 0.1  # below this a price is shown to three decimals
ProviderFactory = Callable[[str, CloudProviderSettings, str], OpenAICompatibleProvider]


@dataclass(frozen=True)
class KeyStatus:
    provider: str
    label: str
    has_key: bool
    active: bool = False
    preset: str = CUSTOM


def _usd(amount: float) -> str:
    return f"${amount:.2f}" if amount >= _CENTS_THRESHOLD_USD else f"${amount:.3f}"


@dataclass(frozen=True)
class CloudModel:
    """One model a provider offers: its id, context length and USD price per million tokens."""

    id: str
    context: int | None = None
    price: tuple[float, float] | None = None  # (input, output)

    @property
    def label(self) -> str:
        """``id · 128k ctx · $0.15 / $0.60 per 1M``, with the parts that are known."""
        parts = [self.id]
        if self.context:
            parts.append(f"{self.context // _TOKENS_PER_K}k ctx")
        if self.price == (0.0, 0.0):
            parts.append("free")
        elif self.price is not None:
            parts.append(f"{_usd(self.price[0])} / {_usd(self.price[1])} per 1M")
        return " · ".join(parts)


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
        make_provider: ProviderFactory = OpenAICompatibleProvider,
        forget_consent: Callable[[], None] = lambda: None,
    ) -> None:
        self._make_provider = make_provider
        self._forget_consent = forget_consent
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
        cloud = self.settings().cloud
        return [
            KeyStatus(
                name,
                cfg.label or name,
                self._keys.get(name) is not None,
                active=cloud.active == name,
                preset=cfg.preset,
            )
            for name, cfg in cloud.providers.items()
        ]

    def add_provider(
        self, preset: str, key: str, base_url: str | None = None, label: str | None = None
    ) -> str:
        """Save a provider from a preset plus its key; returns its name. The first provider added
        becomes the active one. Adding a preset that is already there updates it (new key)."""
        key = key.strip()
        if not key:
            raise SettingsError("enter the API key")
        address = self._address(preset, base_url)
        cloud = self.settings().cloud
        name = preset if preset != CUSTOM else _free_name(CUSTOM, cloud.providers)
        shown = (label or "").strip() or (PRESETS[preset].label if preset != CUSTOM else "")
        base = ["cloud", "providers", name]
        changes: list[tuple[list[str], object]] = [
            ([*base, "base_url"], address),
            ([*base, "preset"], preset),
            ([*base, "label"], shown or None),
        ]
        if cloud.active not in cloud.providers:
            changes.append((["cloud", "active"], name))
        set_settings(self._path, changes)
        self._keys.set(name, key)
        self._on_changed()
        return name

    def remove_provider(self, name: str) -> None:
        """Forget a provider: its key and its settings. The active one falls back to none."""
        cloud = self.settings().cloud
        if name not in cloud.providers:
            raise SettingsError(f"no provider named {name!r}")
        self._keys.delete(name)
        changes: list[tuple[list[str], object]] = [(["cloud", "providers", name], None)]
        if cloud.active == name:
            changes.append((["cloud", "active"], None))
        set_settings(self._path, changes)
        self._on_changed()

    def set_active_provider(self, name: str) -> None:
        self._known(name)
        set_setting(self._path, ["cloud", "active"], name)
        self._on_changed()

    def set_cloud_model(self, name: str, role: str, model: str | None) -> None:
        """Choose the model a provider uses for ``role``; empty clears it (Match: same as chat)."""
        self._known(name)
        if role not in ROLES:
            raise SettingsError(f"unknown role {role!r}")
        set_setting(self._path, ["cloud", "providers", name, "models", role], model or None)
        self._on_changed()

    def set_cloud_routing(self, role: str, policy: str) -> None:
        """When ``role`` uses the cloud: ``local`` (only on "Answer better"), ``auto`` (when the
        local model cannot) or ``cloud`` (always). Ask & Chat carries code questions with it."""
        if role not in ROUTED_ROLES:
            raise SettingsError(f"unknown role {role!r}")
        if policy not in ROUTING_POLICIES:
            raise SettingsError(f"unknown routing {policy!r}")
        stored = None if policy == "local" else policy  # local is the default: leave it unset
        roles = (ROLE_CHAT, ROLE_CODE_CHAT) if role == ROLE_CHAT else (role,)
        set_settings(self._path, [(["cloud", "routing", r], stored) for r in roles])
        self._on_changed()

    def set_fallback_to_local(self, enabled: bool) -> None:
        """Answer with the local model when a routed cloud call fails."""
        set_setting(self._path, ["cloud", "fallback_to_local"], enabled)
        self._on_changed()

    def forget_cloud_consent(self) -> None:
        """Withdraw "don't ask again": the next cloud request asks first."""
        self._forget_consent()

    def set_model_price(self, name: str, model: str, inp: float, out: float) -> None:
        """USD per million input and output tokens, for models whose listing gives no price."""
        self._known(name)
        if inp < 0 or out < 0:
            raise SettingsError("a price cannot be negative")
        set_setting(self._path, ["cloud", "providers", name, "pricing", model], [inp, out])
        self._on_changed()

    def toggle_favorite(self, name: str, model: str) -> bool:
        """Star or unstar ``model`` for the pickers; returns whether it is now a favorite."""
        provider = self._known(name)
        favorites = [m for m in provider.favorites if m != model]
        starred = len(favorites) == len(provider.favorites)
        if starred:
            favorites.append(model)
        set_setting(self._path, ["cloud", "providers", name, "favorites"], favorites or None)
        self._on_changed()
        return starred

    def list_cloud_models(self, name: str) -> list[CloudModel]:
        """The chat models a saved provider offers. Reaches the network: call off the UI thread.
        Errors arrive as ``ProviderError`` with the key already scrubbed out."""
        provider = self._known(name)
        key = self._keys.get(name)
        if not key:
            raise SettingsError(f"no API key stored for {provider.label or name}")
        return self._discover(name, provider, key)

    def probe_provider(
        self, preset: str, key: str, base_url: str | None = None
    ) -> list[CloudModel]:
        """List models with a key that is not saved yet (the add dialog's Test). Network."""
        key = key.strip()
        if not key:
            raise SettingsError("enter the API key")
        settings = CloudProviderSettings(base_url=self._address(preset, base_url), preset=preset)
        return self._discover(preset, settings, key)

    @staticmethod
    def _address(preset: str, base_url: str | None) -> str:
        chosen = PRESETS.get(preset)
        if chosen is None:
            raise SettingsError(f"unknown provider {preset!r}")
        address = chosen.base_url or (base_url or "").strip()
        if not address:
            raise SettingsError("enter the provider's base URL")
        return address

    def _known(self, name: str) -> CloudProviderSettings:
        provider = self.settings().cloud.providers.get(name)
        if provider is None:
            raise SettingsError(f"no provider named {name!r}")
        return provider

    def _discover(self, name: str, cfg: CloudProviderSettings, key: str) -> list[CloudModel]:
        provider = self._make_provider(name, cfg, key)
        return [
            CloudModel(info.name, info.context_length, provider.price(info.name))
            for info in provider.list_models()
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


def _free_name(stem: str, taken: Iterable[str]) -> str:
    used = set(taken)
    candidate, number = stem, 2
    while candidate in used:
        candidate, number = f"{stem}-{number}", number + 1
    return candidate
