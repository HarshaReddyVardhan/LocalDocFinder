"""The editable options of the settings schema, for a settings page generated from it.

Every scalar option (on/off, a choice, a number or text) of every section is listed, so a new
option appears in the Settings window without UI code. Options that a hand-built tab already
edits with its own checks (the hotkey, folders, the cloud, the embedder, the features) are left out.
"""

import types
from dataclasses import dataclass
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from vector_embed.core.settings import Settings, SettingsError

OptionKind = Literal["bool", "choice", "text"]

# Edited elsewhere, with checks of their own; or not meant to be changed by hand at all.
OWNED_SECTIONS = frozenset({"scope", "cloud", "storage", "updates", "features"})
OWNED_OPTIONS = frozenset(
    {
        ("schema_version",),
        ("search", "hotkey"),
        ("app", "start_with_windows"),
        ("app", "theme"),
        ("embedding", "model"),  # switching re-indexes everything: `ve models --embedder`
        ("embedding", "dim"),
        ("privacy", "redact_personal"),
        ("privacy", "mask_ids_locally"),
    }
)
_SCALARS = (bool, int, float, str)
GENERAL_SECTION = "general"


@dataclass(frozen=True)
class OptionSpec:
    """One option: where it lives, how to show it, and how to read typed text into it."""

    path: tuple[str, ...]
    kind: OptionKind
    annotation: Any  # the field's type, used to validate typed text
    choices: tuple[str, ...] = ()
    optional: bool = False  # empty text means "not set" (the default)

    @property
    def section(self) -> str:
        return self.path[0] if len(self.path) > 1 else GENERAL_SECTION

    @property
    def label(self) -> str:
        return self.path[-1].replace("_", " ").capitalize()

    def value_of(self, settings: Settings) -> object:
        node: object = settings
        for key in self.path:
            node = getattr(node, key)
        return node

    def parse(self, text: str) -> object:
        """``text`` as a value of this option's type; ``None`` for an empty optional field."""
        cleaned = text.strip()
        if cleaned == "" and self.optional:
            return None
        try:
            return TypeAdapter(self.annotation).validate_strings(cleaned)
        except ValidationError as exc:
            problem = exc.errors(include_input=False)[0]["msg"]
            raise SettingsError(f"{self.label}: {problem}") from exc


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:  # noqa: ANN401  # a typing form
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1 and len(get_args(annotation)) == 2:  # X | None
            return args[0], True
    return annotation, False


def _spec(path: tuple[str, ...], annotation: Any) -> OptionSpec | None:  # noqa: ANN401
    inner, optional = _unwrap_optional(annotation)
    if get_origin(inner) is Literal:
        values = get_args(inner)
        if all(isinstance(v, str) for v in values):
            return OptionSpec(path, "choice", inner, tuple(values), optional)
        return None
    if inner is bool and not optional:
        return OptionSpec(path, "bool", inner)
    if inner in _SCALARS:
        return OptionSpec(path, "text", inner, optional=optional)
    return None  # lists, mappings, paths and nested models are edited in the TOML file


def _owned(path: tuple[str, ...]) -> bool:
    return path[0] in OWNED_SECTIONS or path in OWNED_OPTIONS


def editable_options(model: type[BaseModel] = Settings) -> list[OptionSpec]:
    """Every scalar option of ``model`` and its sections, in schema order."""
    specs: list[OptionSpec] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            for sub_name, sub_field in annotation.model_fields.items():
                path = (name, sub_name)
                spec = None if _owned(path) else _spec(path, sub_field.annotation)
                if spec is not None:
                    specs.append(spec)
            continue
        spec = None if _owned((name,)) else _spec((name,), annotation)
        if spec is not None:
            specs.append(spec)
    return specs
