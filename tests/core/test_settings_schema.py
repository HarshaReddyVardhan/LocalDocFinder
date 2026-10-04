from typing import Literal

import pytest
from pydantic import BaseModel, Field

from localdoc_finder.core.settings import Settings, SettingsError
from localdoc_finder.core.settings_schema import OWNED_OPTIONS, editable_options


def by_path() -> dict[tuple[str, ...], object]:
    return {option.path: option for option in editable_options()}


def test_every_kind_of_scalar_option_is_listed() -> None:
    options = by_path()
    assert options[("log_level",)].kind == "choice"  # type: ignore[attr-defined]
    assert options[("power", "require_ac_power")].kind == "bool"  # type: ignore[attr-defined]
    assert options[("search", "results")].kind == "text"  # type: ignore[attr-defined]


def test_options_edited_elsewhere_or_not_scalar_are_left_out() -> None:
    paths = set(by_path())
    assert not paths & OWNED_OPTIONS
    assert not any(path[0] in {"scope", "cloud", "storage", "updates"} for path in paths)
    assert ("doctypes", "rules") not in paths  # a mapping: edited in the TOML file


def test_a_new_option_appears_without_ui_code() -> None:
    class Extra(BaseModel):
        speed: int = Field(default=3, gt=0)
        mode: Literal["fast", "slow"] = "fast"
        limit: float | None = None
        tags: tuple[str, ...] = ()

    class Schema(BaseModel):
        extra: Extra = Extra()
        name: str = "x"

    options = {o.path: o for o in editable_options(Schema)}
    assert set(options) == {("extra", "speed"), ("extra", "mode"), ("extra", "limit"), ("name",)}
    assert options[("extra", "limit")].optional
    assert options[("extra", "mode")].choices == ("fast", "slow")
    assert options[("name",)].section == "general"
    assert options[("extra", "speed")].label == "Speed"


def test_typed_text_is_converted_or_rejected_clearly() -> None:
    options = {o.path: o for o in editable_options()}
    results = options[("search", "results")]
    assert results.parse(" 12 ") == 12
    assert results.value_of(Settings()) == Settings().search.results
    with pytest.raises(SettingsError, match="Results"):
        results.parse("many")


def test_an_empty_optional_field_means_not_set() -> None:
    class Schema(BaseModel):
        limit: float | None = None

    (limit,) = editable_options(Schema)
    assert limit.parse("") is None
    assert limit.parse("2.5") == 2.5
